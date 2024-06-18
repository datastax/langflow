import { expect, test } from "@playwright/test";

test("dropDownComponent", async ({ page }) => {
  await page.goto("/");
  await page.waitForTimeout(2000);

  let modalCount = 0;
  try {
    const modalTitleElement = await page?.getByTestId("modal-title");
    if (modalTitleElement) {
      modalCount = await modalTitleElement.count();
    }
  } catch (error) {
    modalCount = 0;
  }

  while (modalCount === 0) {
    await page.getByText("New Project", { exact: true }).click();
    await page.waitForTimeout(5000);
    modalCount = await page.getByTestId("modal-title")?.count();
  }
  await page.waitForTimeout(1000);

  await page.getByTestId("blank-flow").click();
  await page.waitForTimeout(3000);

  await page.getByTestId("extended-disclosure").click();
  await page.getByPlaceholder("Search").click();
  await page.getByPlaceholder("Search").fill("amazon");

  await page.waitForTimeout(1000);

  await page
    .getByTestId("modelsAmazon Bedrock")
    .first()
    .dragTo(page.locator('//*[@id="react-flow-id"]'));
  await page.mouse.up();
  await page.mouse.down();
  await page.getByTitle("fit view").click();
  await page.getByTitle("zoom out").click();
  await page.getByTitle("zoom out").click();
  await page.getByTitle("zoom out").click();
  await page.getByTestId("title-Amazon Bedrock").click();

  await page.getByTestId("dropdown-model_id").click();

  await page.getByTestId("value-dropdown-dropdown-model_id").click();

  let value = await page
    .getByTestId("value-dropdown-dropdown-model_id")
    .first()
    .innerText();
  if (value !== "anthropic.claude-3-haiku-20240307-v1:0") {
    expect(false).toBeTruthy();
  }

  await page.getByTestId("dropdown-model_id").click();
  await page.getByText("anthropic.claude-v2").last().click();

  value = await page.getByTestId("dropdown-model_id").innerText();
  if (value !== "anthropic.claude-v2:1") {
    expect(false).toBeTruthy();
  }

  await page.waitForTimeout(1000);

  await page.getByTestId("more-options-modal").click();
  await page.getByTestId("edit-button-modal").click();

  value = await page.getByTestId("dropdown-edit-model_id").innerText();
  if (value !== "anthropic.claude-v2:1") {
    expect(false).toBeTruthy();
  }

  await page.locator('//*[@id="showcache"]').click();
  expect(await page.locator('//*[@id="showcache"]').isChecked()).toBeFalsy();

  await page.locator('//*[@id="showcache"]').click();
  expect(await page.locator('//*[@id="showcache"]').isChecked()).toBeTruthy();

  await page.locator('//*[@id="showcredentials_profile_name"]').click();
  expect(
    await page.locator('//*[@id="showcredentials_profile_name"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showcredentials_profile_name"]').click();
  expect(
    await page.locator('//*[@id="showcredentials_profile_name"]').isChecked(),
  ).toBeTruthy();

  await page.locator('//*[@id="showendpoint_url"]').click();
  expect(
    await page.locator('//*[@id="showendpoint_url"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showendpoint_url"]').click();
  expect(
    await page.locator('//*[@id="showendpoint_url"]').isChecked(),
  ).toBeTruthy();

  await page.locator('//*[@id="showregion_name"]').click();
  expect(
    await page.locator('//*[@id="showregion_name"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showregion_name"]').click();
  expect(
    await page.locator('//*[@id="showregion_name"]').isChecked(),
  ).toBeTruthy();

  // showmodel_id
  await page.locator('//*[@id="showmodel_id"]').click();
  expect(await page.locator('//*[@id="showmodel_id"]').isChecked()).toBeFalsy();

  // showmodel_id
  await page.locator('//*[@id="showmodel_id"]').click();
  expect(
    await page.locator('//*[@id="showmodel_id"]').isChecked(),
  ).toBeTruthy();

  await page.locator('//*[@id="showcache"]').click();
  expect(await page.locator('//*[@id="showcache"]').isChecked()).toBeFalsy();

  await page.locator('//*[@id="showcache"]').click();
  expect(await page.locator('//*[@id="showcache"]').isChecked()).toBeTruthy();

  await page.locator('//*[@id="showcredentials_profile_name"]').click();
  expect(
    await page.locator('//*[@id="showcredentials_profile_name"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showcredentials_profile_name"]').click();
  expect(
    await page.locator('//*[@id="showcredentials_profile_name"]').isChecked(),
  ).toBeTruthy();

  await page.locator('//*[@id="showendpoint_url"]').click();
  expect(
    await page.locator('//*[@id="showendpoint_url"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showendpoint_url"]').click();
  expect(
    await page.locator('//*[@id="showendpoint_url"]').isChecked(),
  ).toBeTruthy();

  await page.locator('//*[@id="showregion_name"]').click();
  expect(
    await page.locator('//*[@id="showregion_name"]').isChecked(),
  ).toBeFalsy();

  await page.locator('//*[@id="showregion_name"]').click();
  expect(
    await page.locator('//*[@id="showregion_name"]').isChecked(),
  ).toBeTruthy();

  // showmodel_id
  await page.locator('//*[@id="showmodel_id"]').click();
  expect(await page.locator('//*[@id="showmodel_id"]').isChecked()).toBeFalsy();

  // showmodel_id
  await page.locator('//*[@id="showmodel_id"]').click();
  expect(
    await page.locator('//*[@id="showmodel_id"]').isChecked(),
  ).toBeTruthy();

  await page.getByTestId("dropdown-edit-model_id").click();
  await page.getByText("cohere").last().click();

  value = await page.getByTestId("dropdown-edit-model_id").innerText();
  if (value !== "cohere.embed-multilingual-v3") {
    expect(false).toBeTruthy();
  }

  await page.getByText("Save Changes", { exact: true }).click();

  value = await page.getByTestId("dropdown-model_id").innerText();
  if (value !== "cohere.embed-multilingual-v3") {
    expect(false).toBeTruthy();
  }
  await page.getByTestId("code-button-modal").click();
  await page
    .locator("#CodeEditor div")
    .filter({ hasText: "import BedrockChat" })
    .nth(1)
    .click();
  await page.locator("textarea").press("Control+a");
  const emptyOptionsCode = `from langchain_community.chat_models.bedrock import BedrockChat

from langflow.base.constants import STREAM_INFO_TEXT
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import BaseLanguageModel, Text
from langflow.inputs import BoolInput, DictInput, DropdownInput, StrInput
from langflow.inputs import MessageInput
from langflow.template import Output


class AmazonBedrockComponent(LCModelComponent):
    display_name: str = "Amazon Bedrock"
    description: str = "Generate text using Amazon Bedrock LLMs."
    icon = "Amazon"

    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        DropdownInput(
            name="model_id",
            display_name="Model Id",
            options=[
                "amazon.titan-text-express-v1",
                "amazon.titan-text-lite-v1",
                "amazon.titan-text-premier-v1:0",
                "amazon.titan-embed-text-v1",
                "amazon.titan-embed-text-v2:0",
                "amazon.titan-embed-image-v1",
                "amazon.titan-image-generator-v1",
                "anthropic.claude-v2",
                "anthropic.claude-v2:1",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-instant-v1",
                "ai21.j2-mid-v1",
                "ai21.j2-ultra-v1",
                "cohere.command-text-v14",
                "cohere.command-light-text-v14",
                "cohere.command-r-v1:0",
                "cohere.command-r-plus-v1:0",
                "cohere.embed-english-v3",
                "cohere.embed-multilingual-v3",
                "meta.llama2-13b-chat-v1",
                "meta.llama2-70b-chat-v1",
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-70b-instruct-v1:0",
                "mistral.mistral-7b-instruct-v0:2",
                "mistral.mixtral-8x7b-instruct-v0:1",
                "mistral.mistral-large-2402-v1:0",
                "mistral.mistral-small-2402-v1:0",
                "stability.stable-diffusion-xl-v0",
                "stability.stable-diffusion-xl-v1",
            ],
            value="anthropic.claude-3-haiku-20240307-v1:0",
        ),
        StrInput(name="credentials_profile_name", display_name="Credentials Profile Name"),
        StrInput(name="region_name", display_name="Region Name"),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
        StrInput(name="endpoint_url", display_name="Endpoint URL"),
        BoolInput(name="cache", display_name="Cache"),
        StrInput(
            name="system_message",
            display_name="System Message",
            info="System message to pass to the model.",
            advanced=True,
        ),
        BoolInput(name="stream", display_name="Stream", info=STREAM_INFO_TEXT, advanced=True),
    ]
    outputs = [
        Output(display_name="Text", name="text_output", method="text_response"),
        Output(display_name="Language Model", name="model_output", method="build_model"),
    ]

    def text_response(self) -> Text:
        input_value = self.input_value
        stream = self.stream
        system_message = self.system_message
        output = self.build_model()
        result = self.get_chat_result(output, stream, input_value, system_message)
        self.status = result
        return result

    def build_model(self) -> BaseLanguageModel:
        model_id = self.model_id
        credentials_profile_name = self.credentials_profile_name
        region_name = self.region_name
        model_kwargs = self.model_kwargs
        endpoint_url = self.endpoint_url
        cache = self.cache
        stream = self.stream
        try:
            output = BedrockChat(
                credentials_profile_name=credentials_profile_name,
                model_id=model_id,
                region_name=region_name,
                model_kwargs=model_kwargs,
                endpoint_url=endpoint_url,
                streaming=stream,
                cache=cache,
            )
        except Exception as e:
            raise ValueError("Could not connect to AmazonBedrock API.") from e
        return output
  `;
  await page.locator("textarea").fill(emptyOptionsCode);
  await page.getByRole("button", { name: "Check & Save" }).click();
  await page.getByText("No parameters are available for display.").isVisible();
});
