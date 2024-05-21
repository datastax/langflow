import clsx, { ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { priorityFields } from "../constants/constants";
import { ADJECTIVES, DESCRIPTIONS, NOUNS } from "../flow_constants";
import {
  APIDataType,
  APITemplateType,
  TemplateVariableType,
} from "../types/api";
import {
  IVarHighlightType,
  groupedObjType,
  nodeGroupedObjType,
  tweakType,
} from "../types/components";
import { NodeType } from "../types/flow";
import { FlowState } from "../types/tabs";
import { getRandomElement } from "./reactflowUtils";

export function classNames(...classes: Array<string>): string {
  return classes.filter(Boolean).join(" ");
}

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function toNormalCase(str: string): string {
  let result = str
    .split("_")
    .map((word, index) => {
      if (index === 0) {
        return word[0].toUpperCase() + word.slice(1).toLowerCase();
      }
      return word.toLowerCase();
    })
    .join(" ");

  return result
    .split("-")
    .map((word, index) => {
      if (index === 0) {
        return word[0].toUpperCase() + word.slice(1).toLowerCase();
      }
      return word.toLowerCase();
    })
    .join(" ");
}

export function normalCaseToSnakeCase(str: string): string {
  return str
    .split(" ")
    .map((word, index) => {
      if (index === 0) {
        return word[0].toUpperCase() + word.slice(1).toLowerCase();
      }
      return word.toLowerCase();
    })
    .join("_");
}

export function toTitleCase(
  str: string | undefined,
  isNodeField?: boolean,
): string {
  if (!str) return "";
  let result = str
    .split("_")
    .map((word, index) => {
      if (isNodeField) return word;
      if (index === 0) {
        return checkUpperWords(
          word[0].toUpperCase() + word.slice(1).toLowerCase(),
        );
      }
      return checkUpperWords(word.toLowerCase());
    })
    .join(" ");

  return result
    .split("-")
    .map((word, index) => {
      if (isNodeField) return word;
      if (index === 0) {
        return checkUpperWords(
          word[0].toUpperCase() + word.slice(1).toLowerCase(),
        );
      }
      return checkUpperWords(word.toLowerCase());
    })
    .join(" ");
}

export const upperCaseWords: string[] = ["llm", "uri"];
export function checkUpperWords(str: string): string {
  const words = str.split(" ").map((word) => {
    return upperCaseWords.includes(word.toLowerCase())
      ? word.toUpperCase()
      : word[0].toUpperCase() + word.slice(1).toLowerCase();
  });

  return words.join(" ");
}

export function buildInputs(): string {
  return '{"input_value": "message"}';
}

export function getRandomKeyByssmm(): string {
  const now = new Date();
  const seconds = String(now.getSeconds()).padStart(2, "0");
  const milliseconds = String(now.getMilliseconds()).padStart(3, "0");
  return seconds + milliseconds + Math.abs(Math.floor(Math.random() * 10001));
}

export function buildTweakObject(tweak: tweakType) {
  tweak.forEach((el) => {
    Object.keys(el).forEach((key) => {
      for (let kp in el[key]) {
        try {
          el[key][kp] = JSON.parse(el[key][kp]);
        } catch {}
      }
    });
  });
  const tweakString = JSON.stringify(tweak.at(-1), null, 2);
  return tweakString;
}

/**
 * Function to get Chat Input Field
 * @param {FlowsState} tabsState - The current tabs state.
 * @returns {string} - The chat input field
 */
export function getChatInputField(flowState?: FlowState) {
  let chat_input_field = "text";

  if (flowState && flowState.input_keys) {
    chat_input_field = Object.keys(flowState.input_keys!)[0];
  }
  return chat_input_field;
}

export function getOutputIds(flow) {
  const nodes = flow.data!.nodes;

  const arrayOfOutputs = nodes.reduce((acc: string[], node) => {
    if (node.data.type.toLowerCase().includes("output")) {
      acc.push(node.id);
    }
    return acc;
  }, []);

  const arrayOfOutputsJoin = arrayOfOutputs
    .map((output) => `"${output}"`)
    .join(", ");

  return arrayOfOutputsJoin;
}

/**
 * Function to get the widget code for the API
 * @param {string} flow - The current flow.
 * @returns {string} - The widget code
 */
export function getWidgetCode(
  flowId: string,
  flowName: string,
  isAuth: boolean,
): string {
  return `<script src="https://cdn.jsdelivr.net/gh/langflow-ai/langflow-embedded-chat@1.0_alpha/dist/build/static/js/bundle.min.js"></script>

<langflow-chat
  window_title="${flowName}"
  flow_id="${flowId}"
  host_url="http://localhost:7860"${
    !isAuth
      ? `
  api_key="..."`
      : ""
  }

></langflow-chat>`;
}

export function truncateLongId(id: string): string {
  let [componentName, newId] = id.split("-");
  if (componentName.length > 15) {
    componentName = componentName.slice(0, 15);
    componentName += "...";
    return componentName + "-" + newId;
  }
  return id;
}

export function extractIdFromLongId(id: string): string {
  let [_, newId] = id.split("-");
  return newId;
}

export function truncateDisplayName(name: string): string {
  if (name.length > 15) {
    name = name.slice(0, 15);
    name += "...";
  }
  return name;
}

export function tabsArray(codes: string[], method: number) {
  if (!method) return;
  if (method === 0) {
    return [
      {
        name: "cURL",
        mode: "bash",
        image: "https://curl.se/logo/curl-symbol-transparent.png",
        language: "sh",
        code: codes[0],
      },
      {
        name: "Python API",
        mode: "python",
        image:
          "https://images.squarespace-cdn.com/content/v1/5df3d8c5d2be5962e4f87890/1628015119369-OY4TV3XJJ53ECO0W2OLQ/Python+API+Training+Logo.png?format=1000w",
        language: "py",
        code: codes[1],
      },
      {
        name: "Python Code",
        mode: "python",
        image: "https://cdn-icons-png.flaticon.com/512/5968/5968350.png",
        language: "py",
        code: codes[2],
      },
      {
        name: "Chat Widget HTML",
        description:
          "Insert this code anywhere in your &lt;body&gt; tag. To use with react and other libs, check our <a class='link-color' href='https://langflow.org/guidelines/widget'>documentation</a>.",
        mode: "html",
        image: "https://cdn-icons-png.flaticon.com/512/5968/5968350.png",
        language: "py",
        code: codes[3],
      },
    ];
  }
  return [
    {
      name: "cURL",
      mode: "bash",
      image: "https://curl.se/logo/curl-symbol-transparent.png",
      language: "sh",
      code: codes[0],
    },
    {
      name: "Python API",
      mode: "python",
      image:
        "https://images.squarespace-cdn.com/content/v1/5df3d8c5d2be5962e4f87890/1628015119369-OY4TV3XJJ53ECO0W2OLQ/Python+API+Training+Logo.png?format=1000w",
      language: "py",
      code: codes[1],
    },
    {
      name: "Python Code",
      mode: "python",
      language: "py",
      image: "https://cdn-icons-png.flaticon.com/512/5968/5968350.png",
      code: codes[2],
    },
    {
      name: "Chat Widget HTML",
      description:
        "Insert this code anywhere in your &lt;body&gt; tag. To use with react and other libs, check our <a class='link-color' href='https://langflow.org/guidelines/widget'>documentation</a>.",
      mode: "html",
      image: "https://cdn-icons-png.flaticon.com/512/5968/5968350.png",
      language: "py",
      code: codes[3],
    },
    {
      name: "Tweaks",
      mode: "python",
      image: "https://cdn-icons-png.flaticon.com/512/5968/5968350.png",
      language: "py",
      code: codes[4],
    },
  ];
}

export function checkLocalStorageKey(key: string): boolean {
  return localStorage.getItem(key) !== null;
}

export function IncrementObjectKey(
  object: object,
  key: string,
): { newKey: string; increment: number } {
  let count = 1;
  const type = removeCountFromString(key);
  let newKey = type + " " + `(${count})`;
  while (object[newKey]) {
    count++;
    newKey = type + " " + `(${count})`;
  }
  return { newKey, increment: count };
}

export function removeCountFromString(input: string): string {
  // Define a regex pattern to match the count in parentheses
  const pattern = /\s*\(\w+\)\s*$/;

  // Use the `replace` method to remove the matched pattern
  const result = input.replace(pattern, "");

  return result.trim(); // Trim any leading/trailing spaces
}

export function extractTypeFromLongId(id: string): string {
  let [newId, _] = id.split("-");
  return newId;
}

export function createRandomKey(key: string, uid: string): string {
  return removeCountFromString(key) + ` (${uid})`;
}

export function groupByFamily(
  data: APIDataType,
  baseClasses: string,
  left: boolean,
  flow?: NodeType[],
): groupedObjType[] {
  const baseClassesSet = new Set(baseClasses.split("\n"));
  let arrOfPossibleInputs: Array<{
    category: string;
    nodes: nodeGroupedObjType[];
    full: boolean;
    display_name?: string;
  }> = [];
  let arrOfPossibleOutputs: Array<{
    category: string;
    nodes: nodeGroupedObjType[];
    full: boolean;
    display_name?: string;
  }> = [];
  let checkedNodes = new Map();
  const excludeTypes = new Set(["bool", "float", "code", "file", "int"]);

  const checkBaseClass = (template: TemplateVariableType) => {
    return (
      template.type &&
      template.show &&
      ((!excludeTypes.has(template.type) &&
        baseClassesSet.has(template.type)) ||
        (template.input_types &&
          template.input_types.some((inputType) =>
            baseClassesSet.has(inputType),
          )))
    );
  };

  if (flow) {
    // se existir o flow
    for (const node of flow) {
      // para cada node do flow
      if (node!.data!.node!.flow || !node!.data!.node!.template) break; // não faz nada se o node for um group
      const nodeData = node.data;

      const foundNode = checkedNodes.get(nodeData.type); // verifica se o tipo do node já foi checado
      checkedNodes.set(nodeData.type, {
        hasBaseClassInTemplate:
          foundNode?.hasBaseClassInTemplate ||
          Object.values(nodeData.node!.template).some(checkBaseClass),
        hasBaseClassInBaseClasses:
          foundNode?.hasBaseClassInBaseClasses ||
          nodeData.node!.base_classes.some((baseClass) =>
            baseClassesSet.has(baseClass),
          ), //seta como anterior ou verifica se o node tem base class
        displayName: nodeData.node?.display_name,
      });
    }
  }

  for (const [d, nodes] of Object.entries(data)) {
    let tempInputs: nodeGroupedObjType[] = [],
      tempOutputs: nodeGroupedObjType[] = [];

    for (const [n, node] of Object.entries(nodes!)) {
      let foundNode = checkedNodes.get(n);

      if (!foundNode) {
        foundNode = {
          hasBaseClassInTemplate: Object.values(node!.template).some(
            checkBaseClass,
          ),
          hasBaseClassInBaseClasses: node!.base_classes.some((baseClass) =>
            baseClassesSet.has(baseClass),
          ),
          displayName: node?.display_name,
        };
      }

      if (foundNode.hasBaseClassInTemplate)
        tempInputs.push({ node: n, displayName: foundNode.displayName });
      if (foundNode.hasBaseClassInBaseClasses)
        tempOutputs.push({ node: n, displayName: foundNode.displayName });
    }

    const totalNodes = Object.keys(nodes!).length;

    if (tempInputs.length)
      arrOfPossibleInputs.push({
        category: d,
        nodes: tempInputs,
        full: tempInputs.length === totalNodes,
      });
    if (tempOutputs.length)
      arrOfPossibleOutputs.push({
        category: d,
        nodes: tempOutputs,
        full: tempOutputs.length === totalNodes,
      });
  }

  return left
    ? arrOfPossibleOutputs.map((output) => ({
        family: output.category,
        type: output.full
          ? ""
          : output.nodes.map((item) => item.node).join(", "),
        display_name: "",
      }))
    : arrOfPossibleInputs.map((input) => ({
        family: input.category,
        type: input.full ? "" : input.nodes.map((item) => item.node).join(", "),
        display_name: input.nodes.map((item) => item.displayName).join(", "),
      }));
}

// this function is used to get the set of keys from an object
export function getSetFromObject(obj: object, key?: string): Set<string> {
  const set = new Set<string>();
  if (key) {
    for (const objKey in obj) {
      set.add(obj[objKey][key]);
    }
  } else {
    for (const key in obj) {
      set.add(key);
    }
  }
  return set;
}

export function getFieldTitle(
  template: APITemplateType,
  templateField: string,
): string {
  return template[templateField].display_name
    ? template[templateField].display_name!
    : template[templateField].name ?? templateField;
}

export function sortFields(a, b, fieldOrder) {
  // Early return for empty fields
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;

  // Normalize the case to ensure case-insensitive comparison
  const normalizedFieldA = a.toLowerCase();
  const normalizedFieldB = b.toLowerCase();

  const aIsPriority = priorityFields.has(normalizedFieldA);
  const bIsPriority = priorityFields.has(normalizedFieldB);

  // Sort by priority
  if (aIsPriority && !bIsPriority) return -1;
  if (!aIsPriority && bIsPriority) return 1;

  // Check if either field is in the fieldOrder array
  const indexOfA = fieldOrder.indexOf(normalizedFieldA);
  const indexOfB = fieldOrder.indexOf(normalizedFieldB);

  // If both fields are in fieldOrder, sort by their order in the array
  if (indexOfA !== -1 && indexOfB !== -1) {
    return indexOfA - indexOfB;
  }

  // If only one of the fields is in fieldOrder, that field comes first
  if (indexOfA !== -1) {
    return -1;
  }
  if (indexOfB !== -1) {
    return 1;
  }

  // Default case for fields not in priorityFields and not found in fieldOrder
  // You might want to sort them alphabetically or in another specific manner
  return a.localeCompare(b);
}

export function freezeObject(obj: any) {
  if (!obj) return obj;
  return JSON.parse(JSON.stringify(obj));
}

export function convertTestName(name: string): string {
  return name.replace(/ /g, "-").toLowerCase();
}

export function sortByName(stringList: string[]): string[] {
  return stringList.sort((a, b) => a.localeCompare(b));
}
