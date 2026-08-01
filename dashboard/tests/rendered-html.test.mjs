import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("PostCast 대시보드 화면을 서버 렌더링한다", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>PostCast \| 집배 과부하 사전 예보<\/title>/i);
  assert.match(html, /내일의 집배 위험을 계산하고 있습니다/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("분석 데이터와 완성된 화면 구성을 포함한다", async () => {
  const [page, layout, packageJson, dashboardCsv] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(
      new URL("../public/data/district_risk_dashboard.csv", import.meta.url),
      "utf8",
    ),
  ]);

  assert.match(page, /행정동 위험 분포/);
  assert.match(page, /오늘의 대응 우선순위/);
  assert.match(page, /상대 위험지수/);
  assert.match(layout, /PostCast \| 집배 과부하 사전 예보/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(dashboardCsv, /relative_lt_risk_index_base/);

  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
