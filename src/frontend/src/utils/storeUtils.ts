import { cloneDeep } from "lodash";
import { Node } from "reactflow";
import { FlowType, NodeDataType } from "../types/flow";
import { isInputNode, isOutputNode } from "./reactflowUtils";

export default function cloneFLowWithParent(
  flow: FlowType,
  parent: string,
  is_component: boolean
) {
  let childFLow = cloneDeep(flow);
  childFLow.parent = parent;
  childFLow.id = "";
  childFLow.is_component = is_component;
  return childFLow;
}

export function getTagsIds(
  tags: string[],
  tagListId: { name: string; id: string }[]
) {
  return tags
    .map((tag) => tagListId.find((tagObj) => tagObj.name === tag))!
    .map((tag) => tag!.id);
}

export function getInputsAndOutputs(nodes: Node[]) {
  let inputs: { type: string; id: string; displayName: string }[] = [];
  let outputs: { type: string; id: string; displayName: string }[] = [];
  nodes.forEach((node) => {
    const nodeData: NodeDataType = node.data as NodeDataType;
    if (isOutputNode(nodeData)) {
      outputs.push({
        type: nodeData.type,
        id: nodeData.id,
        displayName: nodeData.node?.display_name ?? nodeData.id,
      });
    }
    if (isInputNode(nodeData)) {
      inputs.push({
        type: nodeData.type,
        id: nodeData.id,
        displayName: nodeData.node?.display_name ?? nodeData.id,
      });
    }
  });
  return { inputs, outputs };
}
