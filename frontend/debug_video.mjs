import { chromium } from "playwright";
import path from "node:path";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage();
page.on("console", (m) => console.log("CONSOLE", m.type(), m.text()));
page.on("pageerror", (e) => console.log("PAGEERROR", e.message, e.stack));
page.on("requestfailed", (r) => console.log("REQFAIL", r.url(), r.failure()?.errorText));
await page.goto("http://localhost:5173/video", { waitUntil: "networkidle" });
console.log("title", await page.title());
await page.setInputFiles('input[type="file"]', path.resolve("../../Sample_videos/fire_2.mp4"));
await page.getByRole("button", { name: /run detection/i }).click();
try {
  await page.getByText("first alarm", { exact: false }).waitFor({ timeout: 120000 });
  console.log("RESULT VISIBLE");
} catch (e) {
  console.log("NO RESULT", e.message);
}
console.log("BODY", (await page.locator("body").innerText()).slice(0, 2000));
await page.screenshot({ path: "../artifacts/frontend_video_debug.png", fullPage: true });
await browser.close();
