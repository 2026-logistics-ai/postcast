# PostCast Dashboard

행정 캘린더 기반의 집배 과부하 사전 예보와 행정동별 대응 우선순위를 제공하는 현장 관리자용 웹 대시보드입니다.

## 실행 방법

프로젝트 루트의 `data/processed/district_risk_dashboard.csv`가 준비된 상태에서 실행합니다.

```bash
cd dashboard
npm install
npm run dev
```

`npm run dev`와 `npm run build`를 실행하면 분석 결과 CSV가 `public/data/`로 자동 동기화됩니다.

## 주요 기능

- 예보일·관리 권역·낮음/기준/높음 시나리오 선택
- 예상 배달 물량과 고위험·주의 행정동 집계
- 행정동 중심 좌표 기반 위험 분포 확인
- 날짜별 전체 대응 우선순위 확인
- 선택 행정동의 1인당 부하·거리·면적 점수 분해
- 이벤트 발생에 따른 위험 증가 지역 확인

## 해석 시 주의사항

상대적 LT 지연 위험지수는 실제 작업시간이나 지연 확률이 아닙니다. 평시 기준으로 표준화한 1인당 예상 물량, 거리, 면적을 결합한 대응 우선순위 지표입니다.

## 검증

```bash
npm run build
npm test
```
