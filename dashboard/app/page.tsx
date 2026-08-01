"use client";

import { useEffect, useMemo, useState } from "react";

type Scenario = "low" | "base" | "high";
type DashboardRow = Record<string, string>;

const SCENARIOS: Array<{ value: Scenario; label: string; hint: string }> = [
  { value: "low", label: "낮음", hint: "보수적 물량" },
  { value: "base", label: "기준", hint: "기준 물량" },
  { value: "high", label: "높음", hint: "상향 물량" },
];

const DEFAULT_DATE = "2026-06-10";

function parseCsv(text: string): DashboardRow[] {
  const matrix: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];

    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.length > 0)) matrix.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    matrix.push(row);
  }

  const headers = (matrix.shift() ?? []).map((header, index) =>
    index === 0 ? header.replace(/^\uFEFF/, "") : header,
  );

  return matrix.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

function numberValue(row: DashboardRow | undefined, key: string) {
  if (!row) return 0;
  const value = Number(row[key]);
  return Number.isFinite(value) ? value : 0;
}

function formatNumber(value: number, digits = 0) {
  return new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function formatDate(date: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "long",
    day: "numeric",
    weekday: "short",
  }).format(new Date(`${date}T00:00:00`));
}

function levelClass(level: string) {
  if (level === "High") return "high";
  if (level === "Medium") return "medium";
  return "normal";
}

function levelLabel(level: string) {
  if (level === "High") return "고위험";
  if (level === "Medium") return "주의";
  return "정상";
}

function responseGuide(level: string) {
  if (level === "High") {
    return {
      title: "우선 대응이 필요합니다",
      body: "지원 인력 재배치와 우선 하차·분류 순서 조정을 함께 검토하세요.",
    };
  }
  if (level === "Medium") {
    return {
      title: "작업 여건을 점검하세요",
      body: "예비 인력 가용 여부와 당일 분류 공간을 사전에 확인하세요.",
    };
  }
  return {
    title: "평시 운영이 가능합니다",
    body: "급격한 이벤트 물량 변화가 없는지 접수 현황을 계속 확인하세요.",
  };
}

export default function Home() {
  const [rows, setRows] = useState<DashboardRow[]>([]);
  const [selectedDate, setSelectedDate] = useState(DEFAULT_DATE);
  const [scenario, setScenario] = useState<Scenario>("base");
  const [districtFilter, setDistrictFilter] = useState("전체");
  const [selectedDong, setSelectedDong] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    fetch("/data/district_risk_dashboard.csv")
      .then((response) => {
        if (!response.ok) throw new Error("대시보드 데이터를 불러오지 못했습니다.");
        return response.text();
      })
      .then((text) => {
        if (!active) return;
        const parsedRows = parseCsv(text);
        const availableDates = new Set(parsedRows.map((row) => row.date));
        setRows(parsedRows);
        setSelectedDate(
          availableDates.has(DEFAULT_DATE)
            ? DEFAULT_DATE
            : [...availableDates].sort().at(-1) ?? "",
        );
        setLoading(false);
      })
      .catch((loadError: Error) => {
        if (!active) return;
        setError(loadError.message);
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const dates = useMemo(
    () => [...new Set(rows.map((row) => row.date))].sort(),
    [rows],
  );

  const dateRows = useMemo(
    () => rows.filter((row) => row.date === selectedDate),
    [rows, selectedDate],
  );

  const districts = useMemo(
    () => [...new Set(dateRows.map((row) => row.시구))].sort(),
    [dateRows],
  );

  const visibleRows = useMemo(
    () =>
      districtFilter === "전체"
        ? dateRows
        : dateRows.filter((row) => row.시구 === districtFilter),
    [dateRows, districtFilter],
  );

  const priorityRows = useMemo(
    () =>
      [...visibleRows].sort(
        (a, b) =>
          numberValue(a, `daily_priority_rank_${scenario}`) -
          numberValue(b, `daily_priority_rank_${scenario}`),
      ),
    [visibleRows, scenario],
  );

  useEffect(() => {
    if (!priorityRows.some((row) => row.행정동 === selectedDong)) {
      setSelectedDong(priorityRows[0]?.행정동 ?? "");
    }
  }, [priorityRows, selectedDong]);

  const selectedRow =
    priorityRows.find((row) => row.행정동 === selectedDong) ?? priorityRows[0];

  const eventType = dateRows[0]?.event_type || "없음";
  const planIds = dateRows[0]?.plan_ids || "없음";
  const totalVolume = visibleRows.reduce(
    (sum, row) => sum + numberValue(row, `allocated_volume_${scenario}`),
    0,
  );
  const highRiskCount = visibleRows.filter(
    (row) => row[`relative_risk_level_${scenario}`] === "High",
  ).length;
  const mediumRiskCount = visibleRows.filter(
    (row) => row[`relative_risk_level_${scenario}`] === "Medium",
  ).length;
  const selectedLevel = selectedRow?.[`relative_risk_level_${scenario}`] ?? "Normal";
  const guide = responseGuide(selectedLevel);

  const eventRows = useMemo(
    () =>
      eventType === "없음"
        ? []
        : [...visibleRows]
            .filter((row) => row[`event_uplift_rank_${scenario}`] !== "")
            .sort(
              (a, b) =>
                numberValue(a, `event_uplift_rank_${scenario}`) -
                numberValue(b, `event_uplift_rank_${scenario}`),
            )
            .slice(0, 5),
    [eventType, visibleRows, scenario],
  );

  const mapBounds = useMemo(() => {
    const latitudes = visibleRows.map((row) => numberValue(row, "행정동_위도"));
    const longitudes = visibleRows.map((row) => numberValue(row, "행정동_경도"));
    return {
      minLat: Math.min(...latitudes),
      maxLat: Math.max(...latitudes),
      minLng: Math.min(...longitudes),
      maxLng: Math.max(...longitudes),
    };
  }, [visibleRows]);

  if (loading) {
    return (
      <main className="loading-screen">
        <div className="brand-mark">P</div>
        <p>내일의 집배 위험을 계산하고 있습니다</p>
        <div className="loading-line" />
      </main>
    );
  }

  if (error) {
    return (
      <main className="loading-screen error-screen">
        <div className="brand-mark">!</div>
        <h1>데이터 연결을 확인해주세요</h1>
        <p>{error}</p>
      </main>
    );
  }

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PostCast 대시보드 홈">
          <span className="brand-mark">P</span>
          <span>
            <strong>PostCast</strong>
            <small>Postal overload forecast</small>
          </span>
        </a>
        <div className="status-pill">
          <span className="status-dot" />
          2026 상반기 예보 데이터
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">행정 캘린더 기반 사전 예보</p>
          <h1>
            내일의 집배 과부하,
            <br />오늘 먼저 대응합니다.
          </h1>
          <p className="hero-copy">
            대량 발송계획과 행정동 공간 특성을 결합해 관리자에게 필요한
            대응 우선순위를 한 화면에 제공합니다.
          </p>
        </div>
        <div className="hero-summary">
          <span>선택 예보일</span>
          <strong>{formatDate(selectedDate)}</strong>
          <p className={eventType === "없음" ? "event-none" : "event-active"}>
            {eventType === "없음" ? "등록된 대량 발송 없음" : eventType}
          </p>
        </div>
      </section>

      <section className="control-panel" aria-label="예보 조건 선택">
        <label>
          <span>예보일</span>
          <select
            value={selectedDate}
            onChange={(event) => setSelectedDate(event.target.value)}
          >
            {dates.map((date) => (
              <option key={date} value={date}>
                {date} · {formatDate(date)}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>관리 권역</span>
          <select
            value={districtFilter}
            onChange={(event) => setDistrictFilter(event.target.value)}
          >
            <option value="전체">대전 전체</option>
            {districts.map((district) => (
              <option key={district} value={district}>
                {district}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="scenario-control">
          <legend>물량 시나리오</legend>
          <div>
            {SCENARIOS.map((option) => (
              <button
                type="button"
                key={option.value}
                className={scenario === option.value ? "active" : ""}
                onClick={() => setScenario(option.value)}
                aria-pressed={scenario === option.value}
                title={option.hint}
              >
                {option.label}
              </button>
            ))}
          </div>
        </fieldset>

        <div className="plan-id">
          <span>발송계획 ID</span>
          <strong>{planIds}</strong>
        </div>
      </section>

      <section className="kpi-grid" aria-label="핵심 예보 지표">
        <article className="kpi-card primary">
          <div className="kpi-label">
            <span>예상 배달 물량</span>
            <span className="metric-icon">통</span>
          </div>
          <strong>{formatNumber(totalVolume)}</strong>
          <p>{districtFilter === "전체" ? "대전 전체" : districtFilter} 기준</p>
        </article>
        <article className="kpi-card danger">
          <div className="kpi-label">
            <span>고위험 행정동</span>
            <span className="metric-icon">!</span>
          </div>
          <strong>{highRiskCount}<small>곳</small></strong>
          <p>전체 {visibleRows.length}개 행정동 중</p>
        </article>
        <article className="kpi-card caution">
          <div className="kpi-label">
            <span>주의 행정동</span>
            <span className="metric-icon">△</span>
          </div>
          <strong>{mediumRiskCount}<small>곳</small></strong>
          <p>사전 작업 여건 점검 권장</p>
        </article>
        <article className="kpi-card">
          <div className="kpi-label">
            <span>최우선 대응</span>
            <span className="metric-icon">1</span>
          </div>
          <strong className="dong-name">{priorityRows[0]?.행정동 ?? "-"}</strong>
          <p>{priorityRows[0]?.담당_우체국명 ?? priorityRows[0]?.["담당 우체국명"]}</p>
        </article>
      </section>

      <section className="main-grid">
        <article className="panel map-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">Spatial signal</p>
              <h2>행정동 위험 분포</h2>
            </div>
            <div className="legend" aria-label="위험도 범례">
              <span><i className="normal" />정상</span>
              <span><i className="medium" />주의</span>
              <span><i className="high" />고위험</span>
            </div>
          </div>
          <div className="coordinate-map" aria-label="행정동 좌표 기반 위험 분포도">
            <div className="map-orbit orbit-one" />
            <div className="map-orbit orbit-two" />
            <div className="map-caption">좌표 기반 상대 분포</div>
            {visibleRows.map((row) => {
              const latitude = numberValue(row, "행정동_위도");
              const longitude = numberValue(row, "행정동_경도");
              const left =
                7 +
                ((longitude - mapBounds.minLng) /
                  Math.max(mapBounds.maxLng - mapBounds.minLng, 0.0001)) *
                  86;
              const top =
                7 +
                ((mapBounds.maxLat - latitude) /
                  Math.max(mapBounds.maxLat - mapBounds.minLat, 0.0001)) *
                  86;
              const level = row[`relative_risk_level_${scenario}`];
              const risk = numberValue(row, `relative_lt_risk_index_${scenario}`);
              return (
                <button
                  type="button"
                  key={row.행정동}
                  className={`map-dot ${levelClass(level)} ${
                    selectedRow?.행정동 === row.행정동 ? "selected" : ""
                  }`}
                  style={{ left: `${left}%`, top: `${top}%` }}
                  onClick={() => setSelectedDong(row.행정동)}
                  aria-label={`${row.행정동}, ${levelLabel(level)}, 위험지수 ${risk}`}
                  title={`${row.행정동} · ${levelLabel(level)} ${formatNumber(risk, 1)}`}
                />
              );
            })}
          </div>
          <p className="map-note">
            점 위치는 행정동 중심 좌표이며, 행정경계 면적을 나타내지 않습니다.
          </p>
        </article>

        <article className="panel ranking-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">Action priority</p>
              <h2>오늘의 대응 우선순위</h2>
            </div>
            <span className="scenario-badge">
              {SCENARIOS.find((option) => option.value === scenario)?.label} 시나리오
            </span>
          </div>
          <div className="ranking-list">
            {priorityRows.slice(0, 8).map((row, index) => {
              const level = row[`relative_risk_level_${scenario}`];
              return (
                <button
                  type="button"
                  className={`ranking-row ${
                    selectedRow?.행정동 === row.행정동 ? "selected" : ""
                  }`}
                  key={row.행정동}
                  onClick={() => setSelectedDong(row.행정동)}
                >
                  <span className="rank-number">{index + 1}</span>
                  <span className="rank-place">
                    <strong>{row.행정동}</strong>
                    <small>{row.시구} · {row["담당 우체국명"]}</small>
                  </span>
                  <span className={`risk-chip ${levelClass(level)}`}>
                    {levelLabel(level)}
                  </span>
                  <span className="rank-value">
                    <strong>{formatNumber(numberValue(row, `relative_lt_risk_index_${scenario}`), 1)}</strong>
                    <small>위험지수</small>
                  </span>
                </button>
              );
            })}
          </div>
        </article>
      </section>

      {selectedRow && (
        <section className="detail-grid">
          <article className="panel district-detail">
            <div className="detail-heading">
              <div>
                <p>{selectedRow.시구} · {selectedRow["담당 우체국명"]}</p>
                <h2>{selectedRow.행정동} 상세 진단</h2>
              </div>
              <div className={`risk-score-ring ${levelClass(selectedLevel)}`}>
                <strong>
                  {formatNumber(
                    numberValue(selectedRow, `relative_lt_risk_index_${scenario}`),
                    1,
                  )}
                </strong>
                <span>상대 위험지수</span>
              </div>
            </div>

            <div className="detail-metrics">
              <div>
                <span>예상 물량</span>
                <strong>
                  {formatNumber(numberValue(selectedRow, `allocated_volume_${scenario}`))}
                  <small>통</small>
                </strong>
              </div>
              <div>
                <span>집배원 1인당</span>
                <strong>
                  {formatNumber(numberValue(selectedRow, `volume_per_courier_${scenario}`))}
                  <small>통</small>
                </strong>
              </div>
              <div>
                <span>추정 집배인력</span>
                <strong>
                  {formatNumber(numberValue(selectedRow, "집배원수"), 1)}
                  <small>명</small>
                </strong>
              </div>
            </div>

            <div className="score-breakdown">
              {[
                { label: "1인당 부하", key: "load_score", weight: "70%" },
                { label: "우체국 거리", key: "distance_score", weight: "20%" },
                { label: "배달 면적", key: "area_score", weight: "10%" },
              ].map((metric) => {
                const key = metric.key === "load_score" ? `load_score_${scenario}` : metric.key;
                const score = numberValue(selectedRow, key);
                return (
                  <div className="score-row" key={metric.key}>
                    <div>
                      <span>{metric.label}</span>
                      <small>가중치 {metric.weight}</small>
                    </div>
                    <div className="score-track">
                      <span style={{ width: `${Math.min(score, 100)}%` }} />
                    </div>
                    <strong>{formatNumber(score, 1)}</strong>
                  </div>
                );
              })}
            </div>
          </article>

          <aside className={`response-card ${levelClass(selectedLevel)}`}>
            <span className="response-label">관리자 대응 가이드</span>
            <h2>{guide.title}</h2>
            <p>{guide.body}</p>
            <ul>
              <li>
                <span>전체 대응 순위</span>
                <strong>{formatNumber(numberValue(selectedRow, `daily_priority_rank_${scenario}`))}위</strong>
              </li>
              <li>
                <span>이벤트 추가 물량</span>
                <strong>+{formatNumber(numberValue(selectedRow, "event_added_volume_base"))}통</strong>
              </li>
              <li>
                <span>우체국–행정동 거리</span>
                <strong>{formatNumber(numberValue(selectedRow, "office_dong_distance_km"), 1)}km</strong>
              </li>
            </ul>
            <p className="response-disclaimer">
              현장 기상·결원·당일 접수량을 함께 확인해 최종 대응을 결정하세요.
            </p>
          </aside>
        </section>
      )}

      <section className="bottom-grid">
        <article className="panel uplift-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">Event uplift</p>
              <h2>이벤트로 위험이 커진 지역</h2>
            </div>
            <span className={eventType === "없음" ? "event-tag none" : "event-tag"}>
              {eventType}
            </span>
          </div>
          {eventRows.length > 0 ? (
            <div className="uplift-list">
              {eventRows.map((row) => (
                <button
                  type="button"
                  key={row.행정동}
                  onClick={() => setSelectedDong(row.행정동)}
                >
                  <span className="uplift-rank">
                    {formatNumber(numberValue(row, `event_uplift_rank_${scenario}`))}
                  </span>
                  <span>
                    <strong>{row.행정동}</strong>
                    <small>{row.시구}</small>
                  </span>
                  <span className="uplift-value">
                    +{formatNumber(numberValue(row, `event_risk_delta_${scenario}`), 1)}
                    <small>위험지수</small>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <span>✓</span>
              <div>
                <strong>등록된 대량 발송계획이 없습니다</strong>
                <p>미등록 발송은 예측에 반영되지 않으므로 당일 접수 현황을 확인하세요.</p>
              </div>
            </div>
          )}
        </article>

        <article className="method-card">
          <p className="section-kicker">How to read</p>
          <h2>이 지수는 ‘시간’이 아닌<br />상대적 대응 우선순위입니다.</h2>
          <p>
            평시 집배 환경을 기준으로 1인당 물량·거리·면적을 0–100점으로
            변환해 비교합니다. 실제 퇴근 시각이나 지연 확률을 의미하지 않습니다.
          </p>
          <div className="formula">
            <span><b>70%</b> 1인당 부하</span>
            <i>+</i>
            <span><b>20%</b> 거리</span>
            <i>+</i>
            <span><b>10%</b> 면적</span>
          </div>
        </article>
      </section>

      <footer>
        <div className="brand compact">
          <span className="brand-mark">P</span>
          <strong>PostCast</strong>
        </div>
        <p>행정 캘린더 기반 집배 과부하 사전 예보 · 데이터 분석 및 AI 활용 공모전</p>
        <span>예보 범위 2026.01–06</span>
      </footer>
    </main>
  );
}
