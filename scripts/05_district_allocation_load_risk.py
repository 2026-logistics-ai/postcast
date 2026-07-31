from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("1. 데이터 불러오는 중...")
pred_df = pd.read_csv(
    PROJECT_ROOT / "data/processed/test_plan_stage2_predictions.csv"
)
train_df = pd.read_csv(PROJECT_ROOT / "data/processed/train.csv")
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
bulk_plan = pd.read_csv(
    PROJECT_ROOT / "data/raw/reference/bulk_dispatch_plan.csv"
)

# 컬럼명 공백 제거
dong_biz.columns = dong_biz.columns.str.strip()
dong_info.columns = dong_info.columns.str.strip()
po_coord.columns = po_coord.columns.str.strip()
bulk_plan.columns = bulk_plan.columns.str.strip()


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
        "plan_ids",
        "stage1_baseline_prediction",
        "plan_stage2_adjustment_low",
        "plan_stage2_adjustment_base",
        "plan_stage2_adjustment_high",
        "plan_prediction_low",
        "plan_prediction_base",
        "plan_prediction_high",
    },
    "test_plan_stage2_predictions.csv",
)
validate_required_columns(
    train_df,
    {"접수통수", "is_event", "event_count"},
    "train.csv",
)
validate_required_columns(
    dong_info,
    {
        "행정동명",
        "담당 집배원수(추정)",
        "행정동_위도",
        "행정동_경도",
        "담당 우체국 위도",
        "담당 우체국 경도",
    },
    "행정동별_정보3.csv",
)
validate_required_columns(
    dong_biz,
    {"행정동", "세대수", "사업체수"},
    "행정동별_사업체수_면적_세대_수(csv).csv",
)
validate_required_columns(
    bulk_plan,
    {"계획ID", "이벤트유형"},
    "bulk_dispatch_plan.csv",
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

if bulk_plan["계획ID"].duplicated().any():
    duplicate_plan_ids = sorted(
        bulk_plan.loc[
            bulk_plan["계획ID"].duplicated(keep=False), "계획ID"
        ]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "bulk_dispatch_plan.csv에 중복 계획ID가 있습니다: "
        f"{duplicate_plan_ids}"
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

# 실제 발송 대상의 지역별 구성비 자료가 공개되어 있지 않으므로,
# 세대수와 사업체수를 각각 지역가입자와 사업장가입자의 대체지표로
# 활용한 '세대수·사업체수 기반 공간배분 추정 가중치'를 사용한다.
household_mix_ratio = total_households / (
    total_households + total_businesses
)
business_mix_ratio = total_businesses / (
    total_households + total_businesses
)
dong_biz["weight_mixed"] = (
    household_mix_ratio * dong_biz["weight_household"]
    + business_mix_ratio * dong_biz["weight_business"]
)

event_type_to_weight_column = {
    "주거형": "weight_household",
    "주거형(혼합)": "weight_household",
    "사업체형": "weight_business",
    "혼합형": "weight_mixed",
}
plan_event_type = bulk_plan.set_index("계획ID")[
    "이벤트유형"
].to_dict()

SCENARIOS = ("low", "base", "high")
adjustment_columns = {
    scenario: f"plan_stage2_adjustment_{scenario}"
    for scenario in SCENARIOS
}
prediction_columns = {
    scenario: f"plan_prediction_{scenario}"
    for scenario in SCENARIOS
}

numeric_prediction_columns = [
    "stage1_baseline_prediction",
    *adjustment_columns.values(),
    *prediction_columns.values(),
]
for column in numeric_prediction_columns:
    pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")

if pred_df[numeric_prediction_columns].isna().any().any():
    raise ValueError(
        "test_plan_stage2_predictions.csv의 시나리오 예측 컬럼에 "
        "숫자로 변환할 수 없는 값이 있습니다."
    )

if not (
    (
        pred_df[adjustment_columns["low"]]
        <= pred_df[adjustment_columns["base"]]
    )
    & (
        pred_df[adjustment_columns["base"]]
        <= pred_df[adjustment_columns["high"]]
    )
).all():
    raise ValueError("이벤트 조정 물량이 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다.")

for scenario in SCENARIOS:
    expected_prediction = (
        pred_df["stage1_baseline_prediction"]
        + pred_df[adjustment_columns[scenario]]
    )
    if not np.allclose(
        expected_prediction,
        pred_df[prediction_columns[scenario]],
    ):
        raise ValueError(
            f"{scenario} 시나리오의 최종 예측 물량이 "
            "베이스라인과 이벤트 조정량의 합과 일치하지 않습니다."
        )
    if (expected_prediction < 0).any():
        raise ValueError(
            f"{scenario} 시나리오의 최종 예측 물량에 음수가 있습니다."
        )


def parse_plan_ids(value):
    if pd.isna(value):
        return []

    plan_ids = [
        plan_id.strip()
        for plan_id in str(value).split("|")
        if plan_id.strip() and plan_id.strip() != "없음"
    ]
    return plan_ids

# 3. 행정동별 예측 물량 배분
results = []
for idx, row in pred_df.iterrows():
    date = row["ds"] if "ds" in row else row.get("접수일자", idx)
    baseline_vol = row["stage1_baseline_prediction"]
    event_volumes = {
        scenario: row[adjustment_columns[scenario]]
        for scenario in SCENARIOS
    }
    event_type = str(row.get("active_event_types", "None"))
    active_plan_ids = parse_plan_ids(row.get("plan_ids"))

    if not active_plan_ids:
        nonzero_adjustments = {
            scenario: event_volume
            for scenario, event_volume in event_volumes.items()
            if not np.isclose(event_volume, 0.0)
        }
        if nonzero_adjustments:
            raise ValueError(
                f"{date}: 이벤트 조정 물량은 {nonzero_adjustments}이지만 "
                "plan_ids가 없습니다."
            )
        allocation_event_type = "없음"
        event_weight_column = None
    else:
        if len(active_plan_ids) > 1:
            raise ValueError(
                f"{date}: 복수 계획ID {active_plan_ids}가 존재하지만 "
                "이벤트별 조정 물량이 분리되어 있지 않습니다."
            )

        active_plan_id = active_plan_ids[0]
        if active_plan_id not in plan_event_type:
            raise ValueError(
                f"{date}: bulk_dispatch_plan.csv에 없는 계획ID입니다: "
                f"{active_plan_id}"
            )

        allocation_event_type = plan_event_type[active_plan_id]
        if allocation_event_type not in event_type_to_weight_column:
            raise ValueError(
                f"{date}: 지원하지 않는 이벤트유형입니다: "
                f"{allocation_event_type}"
            )
        event_weight_column = event_type_to_weight_column[
            allocation_event_type
        ]

    for _, dong in dong_biz.iterrows():
        dong_name = dong["행정동"]

        event_weight = (
            dong[event_weight_column]
            if event_weight_column is not None
            else 0.0
        )

        no_event_allocated_volume = (
            baseline_vol * dong["weight_baseline"]
        )
        allocated_volumes = {
            scenario: (
                no_event_allocated_volume
                + event_volumes[scenario] * event_weight
            )
            for scenario in SCENARIOS
        }

        results.append(
            {
                "date": date,
                "행정동": dong_name,
                "event_type": event_type,
                "plan_ids": "|".join(active_plan_ids) or "없음",
                "allocation_event_type": allocation_event_type,
                "no_event_allocated_volume": round(
                    no_event_allocated_volume, 2
                ),
                "allocated_volume_low": round(
                    allocated_volumes["low"], 2
                ),
                "allocated_volume_base": round(
                    allocated_volumes["base"], 2
                ),
                "allocated_volume_high": round(
                    allocated_volumes["high"], 2
                ),
                # 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
                "allocated_volume": round(
                    allocated_volumes["base"], 2
                ),
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

merged_df["면적"] = pd.to_numeric(merged_df["면적"], errors="coerce")
invalid_area_mask = merged_df["면적"].isna() | (merged_df["면적"] <= 0)
if invalid_area_mask.any():
    invalid_dongs = sorted(
        merged_df.loc[invalid_area_mask, "행정동"].astype(str).unique()
    )
    raise ValueError(
        "면적이 누락되었거나 0 이하인 행정동이 있습니다: "
        f"{invalid_dongs}"
    )

# 5. 시나리오별 집배원 1인당 부하량 계산
merged_df["no_event_volume_per_courier"] = np.round(
    merged_df["no_event_allocated_volume"] / merged_df["집배원수"], 2
)

scenario_load_columns = []
scenario_event_added_volume_columns = []
scenario_event_added_load_columns = []
for scenario in SCENARIOS:
    allocated_column = f"allocated_volume_{scenario}"
    load_column = f"volume_per_courier_{scenario}"
    merged_df[load_column] = np.round(
        merged_df[allocated_column] / merged_df["집배원수"], 2
    )
    scenario_load_columns.append(load_column)

    event_added_volume_column = f"event_added_volume_{scenario}"
    event_added_load_column = f"event_added_volume_per_courier_{scenario}"
    merged_df[event_added_volume_column] = np.round(
        merged_df[allocated_column]
        - merged_df["no_event_allocated_volume"],
        2,
    )
    merged_df[event_added_load_column] = np.round(
        merged_df[load_column]
        - merged_df["no_event_volume_per_courier"],
        2,
    )
    scenario_event_added_volume_columns.append(event_added_volume_column)
    scenario_event_added_load_columns.append(event_added_load_column)

# 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
merged_df["volume_per_courier"] = merged_df[
    "volume_per_courier_base"
]

invalid_load_mask = (
    ~np.isfinite(
        merged_df[
            ["no_event_volume_per_courier", *scenario_load_columns]
        ]
    ).all(axis=1)
    | (
        merged_df[
            ["no_event_volume_per_courier", *scenario_load_columns]
        ]
        < 0
    ).any(axis=1)
)
if invalid_load_mask.any():
    invalid_dongs = sorted(
        merged_df.loc[invalid_load_mask, "행정동"].astype(str).unique()
    )
    raise ValueError(
        "집배원 1인당 예상 물량이 유효하지 않은 행정동이 있습니다: "
        f"{invalid_dongs}"
    )

event_added_columns = [
    *scenario_event_added_volume_columns,
    *scenario_event_added_load_columns,
]
if (
    ~np.isfinite(merged_df[event_added_columns]).all(axis=1)
    | (merged_df[event_added_columns] < 0).any(axis=1)
).any():
    raise ValueError(
        "이벤트 추가 물량에 음수 또는 유효하지 않은 값이 있습니다."
    )

for column_prefix in [
    "event_added_volume",
    "event_added_volume_per_courier",
]:
    if not (
        (
            merged_df[f"{column_prefix}_low"]
            <= merged_df[f"{column_prefix}_base"]
        )
        & (
            merged_df[f"{column_prefix}_base"]
            <= merged_df[f"{column_prefix}_high"]
        )
    ).all():
        raise ValueError(
            f"{column_prefix}이 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다."
        )

# 6. 상대적 LT 지연 위험지수 산정을 위한 지리 기초 변수 생성
coordinate_columns = [
    "행정동_위도",
    "행정동_경도",
    "담당 우체국 위도",
    "담당 우체국 경도",
]
for column in coordinate_columns:
    merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")

invalid_coordinate_mask = (
    merged_df[coordinate_columns].isna().any(axis=1)
    | ~merged_df["행정동_위도"].between(-90, 90)
    | ~merged_df["담당 우체국 위도"].between(-90, 90)
    | ~merged_df["행정동_경도"].between(-180, 180)
    | ~merged_df["담당 우체국 경도"].between(-180, 180)
)
if invalid_coordinate_mask.any():
    invalid_dongs = sorted(
        merged_df.loc[invalid_coordinate_mask, "행정동"]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "행정동 또는 담당 우체국 좌표가 유효하지 않은 행정동이 있습니다: "
        f"{invalid_dongs}"
    )


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """두 위경도 지점 사이의 대권거리(km)를 계산한다."""
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    haversine_value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )
    central_angle = 2 * np.arcsin(
        np.sqrt(np.clip(haversine_value, 0.0, 1.0))
    )
    return earth_radius_km * central_angle


def empirical_percentile_score(values, reference_values):
    """참조분포 내 위치를 평균순위 기반 0~100 점수로 변환한다."""
    values_array = np.asarray(values, dtype=float)
    reference_array = np.sort(np.asarray(reference_values, dtype=float))

    if len(reference_array) < 2:
        raise ValueError("백분위 점수 산정을 위한 참조값이 2개 미만입니다.")
    if not np.isfinite(reference_array).all():
        raise ValueError("백분위 점수 참조분포에 유효하지 않은 값이 있습니다.")
    if not np.isfinite(values_array).all():
        raise ValueError("백분위 점수 변환 대상에 유효하지 않은 값이 있습니다.")

    left_rank = np.searchsorted(
        reference_array, values_array, side="left"
    )
    right_rank = np.searchsorted(
        reference_array, values_array, side="right"
    )
    midpoint_rank = (left_rank + right_rank - 1) / 2
    scores = 100 * midpoint_rank / (len(reference_array) - 1)

    scores = np.where(values_array <= reference_array[0], 0.0, scores)
    scores = np.where(values_array >= reference_array[-1], 100.0, scores)
    return np.clip(scores, 0.0, 100.0)


merged_df["office_dong_distance_km"] = np.round(
    haversine_distance_km(
        merged_df["담당 우체국 위도"],
        merged_df["담당 우체국 경도"],
        merged_df["행정동_위도"],
        merged_df["행정동_경도"],
    ),
    3,
)
merged_df["sqrt_area_km"] = np.round(np.sqrt(merged_df["면적"]), 3)

geographic_feature_columns = [
    "office_dong_distance_km",
    "sqrt_area_km",
]
invalid_geographic_feature_mask = (
    ~np.isfinite(merged_df[geographic_feature_columns]).all(axis=1)
    | (merged_df[geographic_feature_columns] < 0).any(axis=1)
)
if invalid_geographic_feature_mask.any():
    invalid_dongs = sorted(
        merged_df.loc[invalid_geographic_feature_mask, "행정동"]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "상대 위험지수 지리 기초 변수가 유효하지 않은 행정동이 있습니다: "
        f"{invalid_dongs}"
    )

# 7. 평시 참조분포 기반 0~100 상대점수 산정
for column in ["접수통수", "is_event", "event_count"]:
    train_df[column] = pd.to_numeric(train_df[column], errors="coerce")

invalid_train_mask = train_df[
    ["접수통수", "is_event", "event_count"]
].isna().any(axis=1)
if invalid_train_mask.any():
    raise ValueError(
        "train.csv의 접수통수 또는 이벤트 판정 컬럼에 "
        "숫자로 변환할 수 없는 값이 있습니다."
    )

normal_train_df = train_df.loc[
    (train_df["is_event"] == 0) & (train_df["event_count"] == 0)
].copy()
if normal_train_df.empty:
    raise ValueError("train.csv에서 평시 참조일을 찾을 수 없습니다.")
if (normal_train_df["접수통수"] < 0).any():
    raise ValueError("평시 참조일의 접수통수에 음수가 있습니다.")

reference_dong_df = pd.merge(
    dong_biz[["행정동", "weight_baseline"]],
    dong_info_clean[["행정동", "집배원수"]],
    on="행정동",
    how="left",
)
reference_dong_df["집배원수"] = pd.to_numeric(
    reference_dong_df["집배원수"], errors="coerce"
)
invalid_reference_dong_mask = (
    reference_dong_df["집배원수"].isna()
    | (reference_dong_df["집배원수"] <= 0)
    | ~np.isfinite(reference_dong_df["weight_baseline"])
    | (reference_dong_df["weight_baseline"] < 0)
)
if invalid_reference_dong_mask.any():
    invalid_dongs = sorted(
        reference_dong_df.loc[invalid_reference_dong_mask, "행정동"]
        .astype(str)
        .unique()
    )
    raise ValueError(
        "평시 부하 참조분포를 만들 수 없는 행정동이 있습니다: "
        f"{invalid_dongs}"
    )

normal_volume_array = normal_train_df["접수통수"].to_numpy(dtype=float)
baseline_load_factor = (
    reference_dong_df["weight_baseline"].to_numpy(dtype=float)
    / reference_dong_df["집배원수"].to_numpy(dtype=float)
)
normal_load_matrix = np.multiply.outer(
    normal_volume_array, baseline_load_factor
)
normal_load_reference = normal_load_matrix.ravel()

spatial_reference_df = merged_df.drop_duplicates("행정동")
if len(spatial_reference_df) != len(dong_biz):
    raise ValueError(
        "공간 참조분포의 행정동 수가 기준정보와 일치하지 않습니다: "
        f"{len(spatial_reference_df)}개 / 기준 {len(dong_biz)}개"
    )

merged_df["no_event_load_score"] = np.round(
    empirical_percentile_score(
        merged_df["no_event_volume_per_courier"],
        normal_load_reference,
    ),
    3,
)

scenario_load_score_columns = []
for scenario in SCENARIOS:
    load_column = f"volume_per_courier_{scenario}"
    load_score_column = f"load_score_{scenario}"
    merged_df[load_score_column] = np.round(
        empirical_percentile_score(
            merged_df[load_column], normal_load_reference
        ),
        3,
    )
    scenario_load_score_columns.append(load_score_column)

# 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
merged_df["load_score"] = merged_df["load_score_base"]
merged_df["distance_score"] = np.round(
    empirical_percentile_score(
        merged_df["office_dong_distance_km"],
        spatial_reference_df["office_dong_distance_km"],
    ),
    3,
)
merged_df["area_score"] = np.round(
    empirical_percentile_score(
        merged_df["sqrt_area_km"],
        spatial_reference_df["sqrt_area_km"],
    ),
    3,
)

score_columns = [
    "no_event_load_score",
    *scenario_load_score_columns,
    "distance_score",
    "area_score",
]
invalid_score_mask = (
    ~np.isfinite(merged_df[score_columns]).all(axis=1)
    | (merged_df[score_columns] < 0).any(axis=1)
    | (merged_df[score_columns] > 100).any(axis=1)
)
if invalid_score_mask.any():
    raise ValueError("0~100 범위를 벗어난 상대점수가 있습니다.")

if not (
    (merged_df["load_score_low"] <= merged_df["load_score_base"])
    & (merged_df["load_score_base"] <= merged_df["load_score_high"])
).all():
    raise ValueError("부하점수가 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다.")

print(
    "상대점수 참조분포 생성 완료: "
    f"평시 {len(normal_train_df)}일 × 행정동 {len(reference_dong_df)}개 "
    f"= {len(normal_load_reference)}건"
)

# 8. 상대적 LT 지연 위험지수 산정
# 실제 LT 시간이나 지연확률이 아니라, 평시 대비 부하와 공간적
# 작업 제약을 결합한 현장 대응 우선순위용 상대지수이다.
LOAD_SCORE_WEIGHT = 0.70
DISTANCE_SCORE_WEIGHT = 0.20
AREA_SCORE_WEIGHT = 0.10

if not np.isclose(
    LOAD_SCORE_WEIGHT + DISTANCE_SCORE_WEIGHT + AREA_SCORE_WEIGHT,
    1.0,
):
    raise ValueError("상대적 LT 지연 위험지수 가중치의 합이 1이 아닙니다.")

scenario_risk_index_columns = []
for scenario in SCENARIOS:
    risk_index_column = f"relative_lt_risk_index_{scenario}"
    merged_df[risk_index_column] = np.round(
        merged_df[f"load_score_{scenario}"] * LOAD_SCORE_WEIGHT
        + merged_df["distance_score"] * DISTANCE_SCORE_WEIGHT
        + merged_df["area_score"] * AREA_SCORE_WEIGHT,
        3,
    )
    scenario_risk_index_columns.append(risk_index_column)

merged_df["no_event_relative_lt_risk_index"] = np.round(
    merged_df["no_event_load_score"] * LOAD_SCORE_WEIGHT
    + merged_df["distance_score"] * DISTANCE_SCORE_WEIGHT
    + merged_df["area_score"] * AREA_SCORE_WEIGHT,
    3,
)

scenario_event_risk_delta_columns = []
for scenario in SCENARIOS:
    event_risk_delta_column = f"event_risk_delta_{scenario}"
    merged_df[event_risk_delta_column] = np.round(
        merged_df[f"relative_lt_risk_index_{scenario}"]
        - merged_df["no_event_relative_lt_risk_index"],
        3,
    )
    scenario_event_risk_delta_columns.append(event_risk_delta_column)

# 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
merged_df["relative_lt_risk_index"] = merged_df[
    "relative_lt_risk_index_base"
]

invalid_relative_index_mask = (
    ~np.isfinite(
        merged_df[
            [
                "no_event_relative_lt_risk_index",
                *scenario_risk_index_columns,
            ]
        ]
    ).all(axis=1)
    | (
        merged_df[
            [
                "no_event_relative_lt_risk_index",
                *scenario_risk_index_columns,
            ]
        ]
        < 0
    ).any(axis=1)
    | (
        merged_df[
            [
                "no_event_relative_lt_risk_index",
                *scenario_risk_index_columns,
            ]
        ]
        > 100
    ).any(axis=1)
)
if invalid_relative_index_mask.any():
    raise ValueError("0~100 범위를 벗어난 상대적 LT 지연 위험지수가 있습니다.")

if not (
    (
        merged_df["relative_lt_risk_index_low"]
        <= merged_df["relative_lt_risk_index_base"]
    )
    & (
        merged_df["relative_lt_risk_index_base"]
        <= merged_df["relative_lt_risk_index_high"]
    )
).all():
    raise ValueError(
        "상대적 LT 지연 위험지수가 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다."
    )

if (
    ~np.isfinite(merged_df[scenario_event_risk_delta_columns]).all(axis=1)
    | (merged_df[scenario_event_risk_delta_columns] < 0).any(axis=1)
).any():
    raise ValueError(
        "이벤트 위험 증가량에 음수 또는 유효하지 않은 값이 있습니다."
    )

if not (
    (merged_df["event_risk_delta_low"] <= merged_df["event_risk_delta_base"])
    & (
        merged_df["event_risk_delta_base"]
        <= merged_df["event_risk_delta_high"]
    )
).all():
    raise ValueError(
        "이벤트 위험 증가량이 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다."
    )

# 기준 시나리오 별칭
merged_df["event_risk_delta"] = merged_df["event_risk_delta_base"]

# 9. 평시 위험지수 분포 기반 상대 위험등급 산정
reference_spatial_score_df = pd.merge(
    reference_dong_df[["행정동"]],
    merged_df[
        ["행정동", "distance_score", "area_score"]
    ].drop_duplicates("행정동"),
    on="행정동",
    how="left",
    validate="one_to_one",
)
if reference_spatial_score_df[
    ["distance_score", "area_score"]
].isna().any().any():
    raise ValueError("평시 위험지수 참조분포의 공간점수가 누락되었습니다.")

normal_load_score_matrix = empirical_percentile_score(
    normal_load_matrix, normal_load_reference
)
normal_relative_risk_reference = (
    normal_load_score_matrix * LOAD_SCORE_WEIGHT
    + reference_spatial_score_df["distance_score"].to_numpy()[None, :]
    * DISTANCE_SCORE_WEIGHT
    + reference_spatial_score_df["area_score"].to_numpy()[None, :]
    * AREA_SCORE_WEIGHT
).ravel()

MEDIUM_RISK_QUANTILE = 0.75
HIGH_RISK_QUANTILE = 0.90
medium_risk_threshold = float(
    np.quantile(normal_relative_risk_reference, MEDIUM_RISK_QUANTILE)
)
high_risk_threshold = float(
    np.quantile(normal_relative_risk_reference, HIGH_RISK_QUANTILE)
)

if not 0 <= medium_risk_threshold < high_risk_threshold <= 100:
    raise ValueError(
        "평시 기준 상대 위험등급 경계값이 유효하지 않습니다: "
        f"Medium={medium_risk_threshold}, High={high_risk_threshold}"
    )

scenario_risk_level_columns = []
for scenario in SCENARIOS:
    risk_index_column = f"relative_lt_risk_index_{scenario}"
    risk_level_column = f"relative_risk_level_{scenario}"
    merged_df[risk_level_column] = np.select(
        [
            merged_df[risk_index_column] >= high_risk_threshold,
            merged_df[risk_index_column] >= medium_risk_threshold,
        ],
        ["High", "Medium"],
        default="Normal",
    )
    scenario_risk_level_columns.append(risk_level_column)

# 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
merged_df["relative_risk_level"] = merged_df[
    "relative_risk_level_base"
]

valid_risk_levels = {"Normal", "Medium", "High"}
calculated_risk_levels = set(
    merged_df[scenario_risk_level_columns].stack().unique()
)
if not calculated_risk_levels.issubset(valid_risk_levels):
    raise ValueError("정의되지 않은 상대 위험등급이 산출되었습니다.")

risk_level_order = {"Normal": 0, "Medium": 1, "High": 2}
low_level_order = merged_df["relative_risk_level_low"].map(
    risk_level_order
)
base_level_order = merged_df["relative_risk_level_base"].map(
    risk_level_order
)
high_level_order = merged_df["relative_risk_level_high"].map(
    risk_level_order
)
if not (
    (low_level_order <= base_level_order)
    & (base_level_order <= high_level_order)
).all():
    raise ValueError("상대 위험등급이 낮음 ≤ 기준 ≤ 높음 순서가 아닙니다.")

print(
    "상대 위험등급 경계 산정 완료: "
    f"Medium ≥ {medium_risk_threshold:.3f}, "
    f"High ≥ {high_risk_threshold:.3f}"
)

# 10. 날짜별 현장 대응 우선순위 산정
if merged_df[["date", "행정동"]].isna().any().any():
    raise ValueError(
        "날짜별 우선순위 산정에 필요한 날짜 또는 행정동이 누락되었습니다."
    )
if merged_df.duplicated(["date", "행정동"]).any():
    raise ValueError(
        "날짜와 행정동 조합이 중복되어 우선순위를 산정할 수 없습니다."
    )

def calculate_daily_priority_rank(df, scenario):
    risk_index_column = f"relative_lt_risk_index_{scenario}"
    load_column = f"volume_per_courier_{scenario}"
    priority_order_index = df.sort_values(
        by=[
            "date",
            risk_index_column,
            load_column,
            "office_dong_distance_km",
            "행정동",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).index
    priority_rank_values = (
        df.loc[priority_order_index]
        .groupby("date", sort=False)
        .cumcount()
        + 1
    )
    priority_rank_by_index = pd.Series(
        priority_rank_values.to_numpy(), index=priority_order_index
    )
    return priority_rank_by_index.reindex(df.index).astype(int)


scenario_priority_rank_columns = []
for scenario in SCENARIOS:
    rank_column = f"daily_priority_rank_{scenario}"
    merged_df[rank_column] = calculate_daily_priority_rank(
        merged_df, scenario
    )
    scenario_priority_rank_columns.append(rank_column)

# 기존 하위 로직과의 호환성을 위한 기준 시나리오 별칭
merged_df["daily_priority_rank"] = merged_df[
    "daily_priority_rank_base"
]

expected_daily_dong_count = len(dong_biz)
for rank_column in scenario_priority_rank_columns:
    daily_rank_validation = merged_df.groupby("date")[rank_column].agg(
        ["count", "nunique", "min", "max"]
    )
    invalid_daily_rank_mask = (
        (daily_rank_validation["count"] != expected_daily_dong_count)
        | (daily_rank_validation["nunique"] != expected_daily_dong_count)
        | (daily_rank_validation["min"] != 1)
        | (daily_rank_validation["max"] != expected_daily_dong_count)
    )
    if invalid_daily_rank_mask.any():
        invalid_dates = daily_rank_validation.index[
            invalid_daily_rank_mask
        ].astype(str).tolist()
        raise ValueError(
            f"{rank_column}에서 1~{expected_daily_dong_count}위가 "
            f"완전하게 생성되지 않은 날짜가 있습니다: {invalid_dates}"
        )

# 11. 이벤트로 추가된 위험 증가량 기준 상승 우선순위 산정
has_event_plan = merged_df["plan_ids"].astype(str).ne("없음")
date_event_status_count = has_event_plan.groupby(merged_df["date"]).nunique()
if (date_event_status_count > 1).any():
    invalid_dates = date_event_status_count.index[
        date_event_status_count > 1
    ].astype(str).tolist()
    raise ValueError(
        "동일 날짜에 이벤트 계획 유무가 혼재되어 있습니다: "
        f"{invalid_dates}"
    )


def calculate_event_uplift_rank(df, scenario, event_mask):
    event_risk_delta_column = f"event_risk_delta_{scenario}"
    event_added_load_column = (
        f"event_added_volume_per_courier_{scenario}"
    )
    risk_index_column = f"relative_lt_risk_index_{scenario}"
    event_df = df.loc[event_mask]
    priority_order_index = event_df.sort_values(
        by=[
            "date",
            event_risk_delta_column,
            event_added_load_column,
            risk_index_column,
            "행정동",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).index
    priority_rank_values = (
        event_df.loc[priority_order_index]
        .groupby("date", sort=False)
        .cumcount()
        + 1
    )
    rank_by_index = pd.Series(
        priority_rank_values.to_numpy(), index=priority_order_index
    )
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    result.loc[priority_order_index] = rank_by_index.astype("Int64")
    return result


scenario_event_uplift_rank_columns = []
for scenario in SCENARIOS:
    rank_column = f"event_uplift_rank_{scenario}"
    merged_df[rank_column] = calculate_event_uplift_rank(
        merged_df, scenario, has_event_plan
    )
    scenario_event_uplift_rank_columns.append(rank_column)

# 기준 시나리오 별칭
merged_df["event_uplift_rank"] = merged_df["event_uplift_rank_base"]

event_dates = merged_df.loc[has_event_plan, "date"].unique()
for rank_column in scenario_event_uplift_rank_columns:
    if merged_df.loc[~has_event_plan, rank_column].notna().any():
        raise ValueError(
            f"{rank_column}에 이벤트가 없는 날짜의 순위가 "
            "생성되었습니다."
        )

    event_rank_validation = (
        merged_df.loc[has_event_plan]
        .groupby("date")[rank_column]
        .agg(["count", "nunique", "min", "max"])
        .reindex(event_dates)
    )
    invalid_event_rank_mask = (
        (event_rank_validation["count"] != expected_daily_dong_count)
        | (event_rank_validation["nunique"] != expected_daily_dong_count)
        | (event_rank_validation["min"] != 1)
        | (event_rank_validation["max"] != expected_daily_dong_count)
    )
    if invalid_event_rank_mask.any():
        invalid_dates = event_rank_validation.index[
            invalid_event_rank_mask
        ].astype(str).tolist()
        raise ValueError(
            f"{rank_column}에서 이벤트 날짜의 1~"
            f"{expected_daily_dong_count}위가 완전하게 생성되지 "
            "않았습니다: "
            f"{invalid_dates}"
        )

print(
    "행정동별 배분 및 시나리오별 상대적 LT 지연 "
    "위험지수·등급·전체 및 이벤트 상승 우선순위 계산 완료!"
)

# 12. 최종 결과 CSV 파일로 저장
output_filename = (
    PROJECT_ROOT
    / "data/processed/district_allocation_load_risk_results.csv"
)
merged_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print("\n모든 행정동 배분 및 상대 위험지수 산정이 완료되었습니다")
print(f"결과 파일 업데이트 완료: {output_filename}")
print("\n--- [이벤트 상승 우선순위 샘플 5건] ---")
print(
    merged_df.loc[has_event_plan].sort_values(
        ["date", "event_uplift_rank_base"]
    )[
        [
            "date",
            "행정동",
            "no_event_allocated_volume",
            "allocated_volume_low",
            "allocated_volume",
            "allocated_volume_high",
            "event_added_volume_low",
            "event_added_volume_base",
            "event_added_volume_high",
            "no_event_volume_per_courier",
            "volume_per_courier_low",
            "volume_per_courier",
            "volume_per_courier_high",
            "event_added_volume_per_courier_low",
            "event_added_volume_per_courier_base",
            "event_added_volume_per_courier_high",
            "office_dong_distance_km",
            "sqrt_area_km",
            "load_score_low",
            "load_score",
            "load_score_high",
            "distance_score",
            "area_score",
            "no_event_relative_lt_risk_index",
            "relative_lt_risk_index_low",
            "relative_lt_risk_index",
            "relative_lt_risk_index_high",
            "event_risk_delta_low",
            "event_risk_delta_base",
            "event_risk_delta_high",
            "relative_risk_level_low",
            "relative_risk_level",
            "relative_risk_level_high",
            "daily_priority_rank_low",
            "daily_priority_rank",
            "daily_priority_rank_high",
            "event_uplift_rank_low",
            "event_uplift_rank_base",
            "event_uplift_rank_high",
        ]
    ].head()
)
