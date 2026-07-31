from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("1. 데이터 불러오는 중...")
pred_df = pd.read_csv(
    PROJECT_ROOT / "data/processed/test_plan_stage2_predictions.csv"
)
dong_info = pd.read_csv(
    PROJECT_ROOT / "data/raw/reference/행정동별_정보3.csv"
)
dong_biz = pd.read_csv(
    PROJECT_ROOT
    / "data/raw/reference/행정동별_사업체수_면적_세대_수(csv).csv"
)
po_coord = pd.read_csv(
    PROJECT_ROOT / "data/raw/reference/대전_통상배달국_좌표.csv"
)

# 컬럼명 공백 제거
dong_biz.columns = dong_biz.columns.str.strip()
dong_info.columns = dong_info.columns.str.strip()
po_coord.columns = po_coord.columns.str.strip()


def validate_required_columns(df, required_columns, dataset_name):
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{dataset_name}에 필수 컬럼이 없습니다: {missing_columns}"
        )


validate_required_columns(
    pred_df,
    {
        "접수일자",
        "active_event_types",
        "stage1_baseline_prediction",
        "plan_stage2_adjustment_base",
    },
    "test_plan_stage2_predictions.csv",
)
validate_required_columns(
    dong_info,
    {"행정동명", "담당 집배원수(추정)"},
    "행정동별_정보3.csv",
)
validate_required_columns(
    dong_biz,
    {"행정동", "세대수", "사업체수"},
    "행정동별_사업체수_면적_세대_수(csv).csv",
)

if dong_info["행정동명"].duplicated().any():
    duplicate_dongs = sorted(
        dong_info.loc[
            dong_info["행정동명"].duplicated(keep=False), "행정동명"
        ]
        .astype(str)
        .unique()
    )
    raise ValueError(
        f"행정동별_정보3.csv에 중복 행정동이 있습니다: {duplicate_dongs}"
    )

if dong_biz["행정동"].duplicated().any():
    duplicate_dongs = sorted(
        dong_biz.loc[
            dong_biz["행정동"].duplicated(keep=False), "행정동"
        ]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "행정동별_사업체수_면적_세대_수(csv).csv에 "
        f"중복 행정동이 있습니다: {duplicate_dongs}"
    )

# '면적' 관련 컬럼명 유연하게 찾기
area_col = [c for c in dong_biz.columns if "면적" in c]
area_col_name = area_col[0] if area_col else None

if area_col_name:
    dong_biz = dong_biz.rename(columns={area_col_name: "면적"})
else:
    dong_biz["면적"] = 1.0

# 2. 행정동별 가중치 산정
total_households = dong_biz["세대수"].sum()
total_businesses = dong_biz["사업체수"].sum()

dong_biz["weight_household"] = dong_biz["세대수"] / total_households
dong_biz["weight_business"] = dong_biz["사업체수"] / total_businesses
dong_biz["weight_baseline"] = (
    dong_biz["weight_household"] + dong_biz["weight_business"]
) / 2

# 3. 행정동별 예측 물량 배분
results = []
for idx, row in pred_df.iterrows():
    date = row["ds"] if "ds" in row else row.get("접수일자", idx)
    baseline_vol = row["stage1_baseline_prediction"]
    event_vol = row["plan_stage2_adjustment_base"]
    event_type = str(row.get("active_event_types", "None"))

    for _, dong in dong_biz.iterrows():
        dong_name = dong["행정동"]

        if "사업소분" in event_type or "사업장" in event_type:
            event_weight = dong["weight_business"]
        else:
            event_weight = dong["weight_household"]

        allocated_vol = (
            baseline_vol * dong["weight_baseline"]
        ) + (event_vol * event_weight)

        results.append(
            {
                "date": date,
                "행정동": dong_name,
                "event_type": event_type,
                "allocated_volume": round(allocated_vol, 2),
            }
        )

allocated_df = pd.DataFrame(results)

# 4. 집배원 정보 및 좌표/면적 결합
dong_info_clean = dong_info.rename(
    columns={
        "행정동명": "행정동",
        "담당 집배원수(추정)": "집배원수",
    }
)

# 사업체수/면적 데이터 병합
merged_df = pd.merge(
    allocated_df, dong_info_clean, on="행정동", how="left"
)
merged_df = pd.merge(
    merged_df, dong_biz[["행정동", "면적"]], on="행정동", how="left"
)

merged_df["집배원수"] = pd.to_numeric(
    merged_df["집배원수"], errors="coerce"
)
invalid_courier_mask = (
    merged_df["집배원수"].isna() | (merged_df["집배원수"] <= 0)
)
if invalid_courier_mask.any():
    invalid_dongs = sorted(
        merged_df.loc[invalid_courier_mask, "행정동"]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "집배원 수가 누락되었거나 0 이하인 행정동이 있습니다: "
        f"{invalid_dongs}"
    )

merged_df["면적"] = pd.to_numeric(
    merged_df["면적"], errors="coerce"
).fillna(1.0)

# 5. 집배원 1인당 부하량 계산
merged_df["volume_per_courier"] = round(
    merged_df["allocated_volume"] / merged_df["집배원수"], 2
)

# 6. LT (Lead Time) 산정 로직 (거리 + 면적 + 물량 기반)
merged_df["travel_time_min"] = round(
    np.sqrt(merged_df["면적"]) * 15, 1
)  # 이동 소요시간(분)
merged_df["delivery_time_min"] = round(
    merged_df["volume_per_courier"] * 1.5, 1
)  # 배달 소요시간(분)
merged_df["total_lt_hours"] = round(
    (
        merged_df["travel_time_min"]
        + merged_df["delivery_time_min"]
    )
    / 60,
    2,
)  # 총 LT(시간)

# 7. 과부하 및 지연 위험도(Risk) 종합 산정
load_threshold = merged_df["volume_per_courier"].quantile(0.90)
lt_threshold = merged_df["total_lt_hours"].quantile(0.90)


def calculate_risk(row):
    if (
        row["volume_per_courier"] >= load_threshold
        or row["total_lt_hours"] >= lt_threshold
    ):
        return "High"
    elif (
        row["volume_per_courier"] >= load_threshold * 0.7
        or row["total_lt_hours"] >= lt_threshold * 0.7
    ):
        return "Medium"
    else:
        return "Normal"


merged_df["overall_risk"] = merged_df.apply(calculate_risk, axis=1)

print("행정동별 배분, 집배원 부하, LT 산정 및 종합 위험도 계산 완료!")

# 8. 최종 결과 CSV 파일로 저장
output_filename = (
    PROJECT_ROOT
    / "data/processed/district_allocation_load_risk_results.csv"
)
merged_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print("\n모든 분석 및 LT 산정이 완료되었습니다")
print(f"결과 파일 업데이트 완료: {output_filename}")
print("\n--- [최종 결과 샘플 5건] ---")
print(
    merged_df[
        [
            "date",
            "행정동",
            "allocated_volume",
            "volume_per_courier",
            "total_lt_hours",
            "overall_risk",
        ]
    ].head()
)
