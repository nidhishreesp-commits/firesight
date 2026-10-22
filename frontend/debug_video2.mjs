import { chromium } from "playwright";
import path from "node:path";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("PAGEERROR", e.message));
page.on("console", (m) => { if (m.type() === "error") console.log("CONSOLE_ERR", m.text()); });
await page.goto("http://localhost:5173/video", { waitUntil: "networkidle" });
await page.setInputFiles('input[type="file"]', path.resolve("../../Sample_videos/fire_2.mp4"));
await page.getByRole("button", { name: /run detection/i }).click();
await page.getByText("first alarm", { exact: false }).waitFor({ timeout: 180000 });
console.log("RESULT_VISIBLE");
const stats = (await page.locator("main").innerText()).slice(0, 600);
console.log("STATS", stats.replace(/\n+/g, " | "));
const video = page.locator("video");
console.log("VIDEO_COUNT", await video.count());
if (await video.count()) {
  await page.evaluate(() => { const v = document.querySelector("video"); if (v) v.currentTime = 10; });
  await page.waitForTimeout(1500);
  const boxes = await page.locator("div.pointer-events-none > div.absolute.border-2").count();
  console.log("OVERLAY_BOXES_AT_10s", boxes);
  await page.screenshot({ path: "../artifacts/frontend_video_preview.png", fullPage: true });
}
await page.getByRole("link", { name: "Test bench" }).click();
await page.waitForTimeout(1000);
await page.getByRole("link", { name: "Video test" }).click();
await page.waitForTimeout(1500);
console.log("AFTER_TAB_SWITCH_first_alarm", (await page.getByText("first alarm", { exact: false }).count()) > 0);
console.log("AFTER_TAB_SWITCH_video", await page.locator("video").count());
const still = (await page.locator("main").innerText()).includes("first alarm");
console.log("AFTER_TAB_SWITCH_stats", still); console.log("VIDEO_DIAG", await page.evaluate(() => { const v=document.querySelector("video"); return v ? {src:v.src, readyState:v.readyState, networkState:v.networkState, currentTime:v.currentTime, error:v.error?.message ?? null, duration:v.duration} : null; }));
await page.screenshot({ path: "../artifacts/frontend_video_after_tab.png", fullPage: true });
await browser.close();

