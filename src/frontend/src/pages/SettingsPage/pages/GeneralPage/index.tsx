import * as Form from "@radix-ui/react-form";
import { useContext, useState } from "react";
import { useParams } from "react-router-dom";
import ForwardedIconComponent from "../../../../components/genericIconComponent";
import GradientChooserComponent from "../../../../components/gradientChooserComponent";
import InputComponent from "../../../../components/inputComponent";
import { Button } from "../../../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../../../../components/ui/card";
import {
  CONTROL_PATCH_USER_STATE,
  CREATE_API_KEY,
  INSERT_API_KEY,
  INVALID_API_KEY,
  NO_API_KEY,
} from "../../../../constants/constants";
import { AuthContext } from "../../../../contexts/authContext";
import useAlertStore from "../../../../stores/alertStore";
import useFlowsManagerStore from "../../../../stores/flowsManagerStore";
import { useStoreStore } from "../../../../stores/storeStore";
import {
  inputHandlerEventType,
  patchUserInputStateType,
} from "../../../../types/components";
import { gradients } from "../../../../utils/styleUtils";
import usePatchGradient from "../hooks/use-patch-gradient";
import usePatchPassword from "../hooks/use-patch-password";
import useSaveKey from "../hooks/use-save-key";
import useScrollToElement from "../hooks/use-scroll-to-element";

export default function GeneralPage() {
  const setCurrentFlowId = useFlowsManagerStore(
    (state) => state.setCurrentFlowId
  );

  const { scrollId } = useParams();

  const [inputState, setInputState] = useState<patchUserInputStateType>(
    CONTROL_PATCH_USER_STATE
  );

  const { autoLogin } = useContext(AuthContext);

  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const { userData, setUserData } = useContext(AuthContext);
  const hasStore = useStoreStore((state) => state.hasStore);

  const validApiKey = useStoreStore((state) => state.validApiKey);
  const hasApiKey = useStoreStore((state) => state.hasApiKey);
  const setHasApiKey = useStoreStore((state) => state.updateHasApiKey);
  const loadingApiKey = useStoreStore((state) => state.loadingApiKey);
  const setValidApiKey = useStoreStore((state) => state.updateValidApiKey);
  const setLoadingApiKey = useStoreStore((state) => state.updateLoadingApiKey);
  const { password, cnfPassword, gradient, apikey } = inputState;

  const { handlePatchPassword } = usePatchPassword(
    userData,
    setSuccessData,
    setErrorData
  );

  const { handlePatchGradient } = usePatchGradient(
    setSuccessData,
    setErrorData,
    userData,
    setUserData
  );

  useScrollToElement(scrollId, setCurrentFlowId);

  const { handleSaveKey } = useSaveKey(
    setSuccessData,
    setErrorData,
    setHasApiKey,
    setValidApiKey,
    setLoadingApiKey
  );

  function handleInput({
    target: { name, value },
  }: inputHandlerEventType): void {
    setInputState((prev) => ({ ...prev, [name]: value }));
  }

  return (
    <div className="flex h-full w-full flex-col gap-6">
      <div className="flex w-full items-center justify-between gap-4 space-y-0.5">
        <div className="flex w-full flex-col">
          <h2 className="flex items-center text-lg font-semibold tracking-tight">
            General
            <ForwardedIconComponent
              name="SlidersHorizontal"
              className="ml-2 h-5 w-5 text-primary"
            />
          </h2>
          <p className="text-sm text-muted-foreground">
            Manage settings related to Langflow and your account.
          </p>
        </div>
      </div>

      <div className="grid gap-6">
        <Form.Root
          onSubmit={(event) => {
            handlePatchGradient(gradient);
            event.preventDefault();
          }}
        >
          <Card x-chunk="dashboard-04-chunk-1">
            <CardHeader>
              <CardTitle>Profile Gradient</CardTitle>
              <CardDescription>
                Choose the gradient that appears as your profile picture.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="py-2">
                <GradientChooserComponent
                  value={
                    gradient == ""
                      ? userData?.profile_image ??
                        gradients[
                          parseInt(userData?.id ?? "", 30) % gradients.length
                        ]
                      : gradient
                  }
                  onChange={(value) => {
                    handleInput({ target: { name: "gradient", value } });
                  }}
                />
              </div>
            </CardContent>
            <CardFooter className="border-t px-6 py-4">
              <Form.Submit asChild>
                <Button type="submit">Save</Button>
              </Form.Submit>
            </CardFooter>
          </Card>
        </Form.Root>
        {!autoLogin && (
          <Form.Root
            onSubmit={(event) => {
              handlePatchPassword(password, cnfPassword, handleInput);
              event.preventDefault();
            }}
          >
            <Card x-chunk="dashboard-04-chunk-2">
              <CardHeader>
                <CardTitle>Password</CardTitle>
                <CardDescription>
                  Type your new password and confirm it.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex w-full gap-4">
                  <Form.Field name="password" className="w-full">
                    <InputComponent
                      id="pasword"
                      onChange={(value) => {
                        handleInput({ target: { name: "password", value } });
                      }}
                      value={password}
                      isForm
                      password={true}
                      placeholder="Password"
                      className="w-full"
                    />
                    <Form.Message
                      match="valueMissing"
                      className="field-invalid"
                    >
                      Please enter your password
                    </Form.Message>
                  </Form.Field>
                  <Form.Field name="cnfPassword" className="w-full">
                    <InputComponent
                      id="cnfPassword"
                      onChange={(value) => {
                        handleInput({
                          target: { name: "cnfPassword", value },
                        });
                      }}
                      value={cnfPassword}
                      isForm
                      password={true}
                      placeholder="Confirm Password"
                      className="w-full"
                    />

                    <Form.Message
                      className="field-invalid"
                      match="valueMissing"
                    >
                      Please confirm your password
                    </Form.Message>
                  </Form.Field>
                </div>
              </CardContent>
              <CardFooter className="border-t px-6 py-4">
                <Form.Submit asChild>
                  <Button type="submit">Save</Button>
                </Form.Submit>
              </CardFooter>
            </Card>
          </Form.Root>
        )}
        {hasStore && (
          <Form.Root
            onSubmit={(event) => {
              event.preventDefault();
              handleSaveKey(apikey, handleInput);
            }}
          >
            <Card x-chunk="dashboard-04-chunk-2" id="api">
              <CardHeader>
                <CardTitle>Store API Key</CardTitle>
                <CardDescription>
                  {(hasApiKey && !validApiKey
                    ? INVALID_API_KEY
                    : !hasApiKey
                    ? NO_API_KEY
                    : "") + INSERT_API_KEY}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex w-full flex-col gap-3">
                  <div className="flex w-full gap-4">
                    <Form.Field name="apikey" className="w-full">
                      <InputComponent
                        id="apikey"
                        onChange={(value) => {
                          handleInput({ target: { name: "apikey", value } });
                        }}
                        value={apikey}
                        isForm
                        password={true}
                        placeholder="Insert your API Key"
                        className="w-full"
                      />
                      <Form.Message
                        match="valueMissing"
                        className="field-invalid"
                      >
                        Please enter your API Key
                      </Form.Message>
                    </Form.Field>
                  </div>
                  <span className="pr-1 text-xs text-muted-foreground">
                    {CREATE_API_KEY}{" "}
                    <a
                      className="text-high-indigo underline"
                      href="https://langflow.store/"
                      target="_blank"
                    >
                      langflow.store
                    </a>
                  </span>
                </div>
              </CardContent>
              <CardFooter className="border-t px-6 py-4">
                <Form.Submit asChild>
                  <Button
                    loading={loadingApiKey}
                    type="submit"
                    data-testid="api-key-save-button-store"
                  >
                    Save
                  </Button>
                </Form.Submit>
              </CardFooter>
            </Card>
          </Form.Root>
        )}
      </div>
    </div>
  );
}
