import { ColDef } from "ag-grid-community";
import { forwardRef, useEffect, useRef } from "react";
import IconComponent from "../../components/genericIconComponent";
import TableComponent from "../../components/tableComponent";
import { Badge } from "../../components/ui/badge";
import useFlowStore from "../../stores/flowStore";
import { NodeDataType } from "../../types/flow";
import BaseModal from "../baseModal";
import useColumnDefs from "./hooks/use-column-defs";
import useRowData from "./hooks/use-row-data";

const EditNodeModal = forwardRef(
  (
    {
      nodeLength,
      open,
      setOpen,
      data,
    }: {
      nodeLength: number;
      open: boolean;
      setOpen: (open: boolean) => void;
      data: NodeDataType;
    },
    ref
  ) => {
    const nodes = useFlowStore((state) => state.nodes);

    const dataFromStore = nodes.find((node) => node.id === node.id)?.data;

    const myData = useRef(dataFromStore ?? data);

    const setNode = useFlowStore((state) => state.setNode);

    function changeAdvanced(n) {
      myData.current.node!.template[n].advanced =
        !myData.current.node!.template[n]?.advanced;
    }

    const handleOnNewValue = (newValue: any, name) => {
      myData.current.node!.template[name].value = newValue;
    };

    useEffect(() => {
      if (open) {
        myData.current = data;
      }
    }, [open]);

    const rowData = useRowData(myData);

    const columnDefs: ColDef[] = useColumnDefs(
      myData,
      handleOnNewValue,
      changeAdvanced
    );

    return (
      <BaseModal
        key={data.id}
        size="large-h-full"
        open={open}
        setOpen={setOpen}
        onChangeOpenModal={(open) => {
          myData.current = data;
        }}
        onSubmit={() => {
          setNode(data.id, (old) => ({
            ...old,
            data: {
              ...old.data,
              node: myData.current.node,
            },
          }));
          setOpen(false);
        }}
      >
        <BaseModal.Trigger>
          <></>
        </BaseModal.Trigger>
        <BaseModal.Header description={myData.current.node?.description!}>
          <span className="pr-2">{myData.current.type}</span>
          <Badge variant="secondary">ID: {myData.current.id}</Badge>
        </BaseModal.Header>
        <BaseModal.Content>
          <div className="flex pb-2">
            <IconComponent
              name="Variable"
              className="edit-node-modal-variable "
            />
            <span className="edit-node-modal-span">Parameters</span>
          </div>

          <div className="w-full">
            {nodeLength > 0 && (
              <div className="edit-node-modal-table">
                <div className="h-96">
                  <TableComponent
                    tooltipShowMode="whenTruncated"
                    tooltipShowDelay={0.5}
                    columnDefs={columnDefs}
                    rowData={rowData}
                  />
                </div>
              </div>
            )}
          </div>
        </BaseModal.Content>

        <BaseModal.Footer submit={{ label: "Save Changes" }} />
      </BaseModal>
    );
  }
);

export default EditNodeModal;
