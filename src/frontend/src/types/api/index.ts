import { Edge, Node, Viewport } from "reactflow";
import { FlowType } from "../flow";
//kind and class are just representative names to represent the actual structure of the object received by the API
export type APIDataType = { [key: string]: APIKindType };
export type APIObjectType = { kind: APIKindType; [key: string]: APIKindType };
export type APIKindType = { class: APIClassType; [key: string]: APIClassType };
export type APITemplateType = {
  [key: string]: TemplateVariableType;
};
export type APIClassType = {
  base_classes: Array<string>;
  description: string;
  template: APITemplateType;
  display_name: string;
  input_types?: Array<string>;
  output_types?: Array<string>;
  beta?: boolean;
  documentation: string;
  error?: string;
  flow?:FlowType;
  [key: string]: Array<string> | string | APITemplateType | boolean | FlowType | undefined;
};

export type TemplateVariableType = {
  type: string;
  required: boolean;
  placeholder?: string;
  list: boolean;
  show: boolean;
  multiline?: boolean;
  value?: any;
  input_types?: Array<string>;
  [key: string]: any;
};
export type sendAllProps = {
  nodes: Node[];
  edges: Edge[];
  name: string;
  description: string;
  viewport: Viewport;
  inputs: { text?: string };

  chatHistory: { message: string | object; isSend: boolean }[];
};
export type errorsTypeAPI = {
  function: { errors: Array<string> };
  imports: { errors: Array<string> };
};
export type PromptTypeAPI = {
  input_variables: Array<string>;
  frontend_node: APIClassType;
};

export type BuildStatusTypeAPI = {
  built: boolean;
};

export type InitTypeAPI = {
  flowId: string;
};

export type UploadFileTypeAPI = {
  file_path: string;
  flowId: string;
};

export type LoginType = {
  grant_type?: string;
  username: string;
  password: string;
  scrope?: string;
  client_id?: string;
  client_secret?: string;
};

export type LoginAuthType = {
  access_token: string;
  refresh_token: string;
  token_type?: string;
};

export type Users = {
  id: string;
  username: string;
  is_active: boolean;
  is_superuser: boolean;
  create_at: Date;
  updated_at: Date;
};
