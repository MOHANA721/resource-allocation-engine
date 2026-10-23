import { expect, test } from "@playwright/test";

const GREEDY = 0;
const HUNGARIAN = 1;
const REGRET = 2;

async function loadTrap(page) {
  await page.goto("/");
  await expect(page.locator(".leaflet-container")).toHaveCount(3);
  await page.getByRole("button", { name: "Load trap scenario" }).click();
  await expect(page.locator(".scenario-note")).toContainText("Trap");
}

test("trap scenario: greedy strands the electrician, Hungarian and regret serve both", async ({ page }) => {
  await loadTrap(page);
  const heads = page.locator(".panel-head");
  await expect(heads.nth(GREEDY)).toContainText("150.3 cost");
  await expect(heads.nth(HUNGARIAN)).toContainText("6.3 cost");
  await expect(heads.nth(REGRET)).toContainText("6.3 cost");

  await page.getByRole("button", { name: /Household panel/ }).click();
  const reasons = page.locator(".reason");
  await expect(reasons.nth(GREEDY)).toContainText("every technician with electrical is already full");
  await expect(reasons.nth(HUNGARIAN)).toContainText("Vikram takes this job");
  await expect(reasons.nth(REGRET)).toContainText("largest regret");
});

test("priority-5 checkbox decides whether Hungarian may drop the panel", async ({ page }) => {
  await loadTrap(page);
  await page.getByLabel(/Distance weight/).fill("5");
  await page.getByLabel(/Unassigned weight/).fill("5");
  await page.getByRole("button", { name: /Household panel/ }).click();
  await expect(page.locator(".panel-head").nth(GREEDY)).toContainText("26.3 cost");
  const hungarian = page.locator(".reason").nth(HUNGARIAN);
  await expect(hungarian).toContainText("Vikram takes this job");

  await page.getByLabel("Always serve P5 when possible").uncheck();
  await expect(hungarian).toContainText("left open in the minimum-cost plan");
  await expect(page.locator(".panel-head").nth(HUNGARIAN)).toContainText("26.3 cost");
});

test("working-minutes capacity reallocates without errors", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".leaflet-container")).toHaveCount(3);
  await page.getByLabel("Capacity").selectOption("minutes");
  await expect(page.locator(".loading")).toHaveCount(0);
  await expect(page.locator(".error")).toHaveCount(0);
  await expect(page.locator("table.metrics tbody tr")).toHaveCount(12);
});
