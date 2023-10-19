import { expect, test } from "@playwright/test";
import { readFileSync } from "fs";

test.describe("save component tests", () => {
  /// <reference lib="dom"/>
  test("save group component tests", async ({ page }) => {
    await page.routeFromHAR("harFiles/langflow.har", {
      url: "**/api/v1/**",
      update: false,
    });
    await page.route("**/api/v1/flows/", async (route) => {
      const json = {
        id: "e9ac1bdc-429b-475d-ac03-d26f9a2a3210",
      };
      await route.fulfill({ json, status: 201 });
    });
    await page.goto("http:localhost:3000/");
    await page.locator("span").filter({ hasText: "My Collection" }).isVisible();
    // Read your file into a buffer.
    const jsonContent = readFileSync(
      "tests/onlyFront/assets/flow.json",
      "utf-8"
    );

    // Create the DataTransfer and File
    const dataTransfer = await page.evaluateHandle((data) => {
      const dt = new DataTransfer();
      // Convert the buffer to a hex array
      const file = new File([data], "flow.json", {
        type: "application/json",
      });
      dt.items.add(file);
      return dt;
    }, jsonContent);

    // Now dispatch
    await page.dispatchEvent('//*[@id="root"]/div/div[2]/div[2]', "drop", {
      dataTransfer,
    });
    expect(
      await page
        .locator(".main-page-flows-display")
        .evaluate((el) => el.children)
    ).toBeTruthy();
    await page.getByRole("button", { name: "Edit Flow" }).click();
    //inside the flow
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[1]/div/div[2]/div[1]/div/div[1]/div"
      )
      .click({
        modifiers: ["Control"],
      });
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[1]/div/div[2]/div[2]/div/div[1]/div"
      )
      .click({
        modifiers: ["Control"],
      });
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[1]/div/div[2]/div[3]/div/div[1]/div"
      )
      .click({
        modifiers: ["Control"],
      });
    await page.getByRole("button", { name: "Group" }).click();
    expect(
      await page
        .locator(
          "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[1]/div/div[2]/div/div"
        )
        .isVisible()
    ).toBeTruthy();
    await page.getByPlaceholder("Type something...").first().click();
    await page.getByPlaceholder("Type something...").first().fill("save");
    await page.locator(".react-flow__pane").click();
    await page
      .locator(".side-bar-buttons-arrangement > div:nth-child(3)")
      .click();
    //more option click
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[2]/div/span/button[3]/div/div"
      )
      .click();
    await page.getByLabel("Save").click();
    await page.getByPlaceholder("Search").click();
    await page.getByPlaceholder("Search").fill("save");
    await page.waitForTimeout(2000);
    await page
      .locator('//*[@id="custom_componentssave"]')
      .dragTo(page.locator('//*[@id="react-flow-id"]'));
    await page.waitForTimeout(2000);
    expect(
      (await page.getByTestId(/.*rf__node-AgentInitializer.*/).all()).length
    ).toBe(2);
    await page.locator(".isolate > button").first().click();
    expect(
      (await page.getByTestId(/.*rf__node-AgentInitializer.*/).all()).length
    ).toBe(1);
    await page.getByTestId(/.*rf__node-AgentInitializer.*/).click();
    await page.getByTestId(/.*rf__node-AgentInitializer.*/).press("Backspace");
    await page
      .locator('//*[@id="custom_componentssave"]')
      .dragTo(page.locator('//*[@id="react-flow-id"]'));
    await page.getByTestId(/.*rf__node-AgentInitializer.*/).click();
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[2]/div/span/button[3]/div/div"
      )
      .click();
    await page.getByLabel("Ungroup").click();
    expect((await page.getByTestId(/.*rf__node-.*/).all()).length).toBe(3);
    expect(
      (await page.getByTestId(/.*rf__edge-reactflow.*/).all()).length
    ).toBe(2);
  });

  test("save default component with custom values", async ({ page }) => {
    await page.routeFromHAR("harFiles/langflow.har", {
      url: "**/api/v1/**",
      update: false,
    });
    await page.route("**/api/v1/flows/", async (route) => {
      const json = {
        id: "e9ac1bdc-429b-475d-ac03-d26f9a2a3210",
      };
      await route.fulfill({ json, status: 201 });
    });
    await page.goto("http://localhost:3000/");
    await page.locator("span").filter({ hasText: "My Collection" }).isVisible();
    await page.locator('//*[@id="new-project-btn"]').click();
    await page.waitForTimeout(2000);

    await page.getByPlaceholder("Search").click();
    await page.getByPlaceholder("Search").fill("Chroma");

    await page
      .locator('//*[@id="vectorstoresChroma"]')
      .dragTo(page.locator('//*[@id="react-flow-id"]'));
    await page.locator("#input-8").click();
    await page.locator("#input-8").fill("test");

    await page.getByTestId(/.*rf__node-Chroma.*/).click();
    //more node options
    await page
      .locator(
        "//html/body/div/div/div[2]/div/main/div/div/div/div[1]/div[1]/div[2]/div/span/button[3]/div/div"
      )
      .click();
    await page.getByLabel("Save").click();
    await page.getByTestId(/.*rf__node-Chroma.*/).press("Backspace");
    await page.getByPlaceholder("Search").click();
    await page.getByPlaceholder("Search").fill("");
    await page.getByPlaceholder("Search").fill("Chroma");
    await page
      .locator('//*[@id="custom_componentsChroma"]')
      .dragTo(page.locator('//*[@id="react-flow-id"]'));
    expect(await page.locator("#input-8").inputValue()).toBe("test");
  });
});
