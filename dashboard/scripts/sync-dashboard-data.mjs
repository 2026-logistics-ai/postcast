import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const source = resolve(
  process.cwd(),
  "../data/processed/district_risk_dashboard.csv",
);
const target = resolve(
  process.cwd(),
  "public/data/district_risk_dashboard.csv",
);

if (existsSync(source)) {
  mkdirSync(dirname(target), { recursive: true });
  copyFileSync(source, target);
  console.log("대시보드 데이터를 최신 분석 결과로 동기화했습니다.");
} else if (!existsSync(target)) {
  throw new Error(
    "district_risk_dashboard.csv를 찾을 수 없습니다. 분석 결과 파일을 먼저 생성해주세요.",
  );
}
