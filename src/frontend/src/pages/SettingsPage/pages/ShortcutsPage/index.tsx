import { ColDef, ColGroupDef, SelectionChangedEvent } from "ag-grid-community";
import { useEffect, useState } from "react";
import ForwardedIconComponent from "../../../../components/genericIconComponent";
import TableComponent from "../../../../components/tableComponent";
import { Button } from "../../../../components/ui/button";
import {
  defaultShortcuts,
  unavailableShortcutss,
} from "../../../../constants/constants";
import { useShortcutsStore } from "../../../../stores/shortcuts";
import EditShortcutButton from "./EditShortcutButton";

export default function ShortcutsPage() {
  const [selectedRows, setSelectedRows] = useState<string[]>([]);
  const shortcuts = useShortcutsStore((state) => state.shortcuts);
  const setShortcuts = useShortcutsStore((state) => state.setShortcuts);

  // Column Definitions: Defines the columns to be displayed.
  const colDefs = [
    {
      headerName: "Functionality",
      field: "name",
      flex: 1,
      editable: false,
      headerCheckboxSelection: true,
      checkboxSelection: true,
      showDisabledCheckboxes: true,
      resizable: false,
    }, //This column will be twice as wide as the others
    {
      field: "shortcut",
      flex: 2,
      editable: false,
      resizable: false,
    },
  ];

  const [nodesRowData, setNodesRowData] = useState<
    Array<{ name: string; shortcut: string }>
  >([]);

  useEffect(() => {
    setNodesRowData(shortcuts);
  }, [shortcuts]);

  const combinationToEdit = shortcuts.filter((s) => s.name === selectedRows[0]);
  const [open, setOpen] = useState(false);

  function handleRestore() {
    setShortcuts(defaultShortcuts, unavailableShortcutss);
    localStorage.removeItem("langflow-shortcuts");
    localStorage.removeItem("langflow-UShortcuts");
  }

  return (
    <div className="flex h-full w-full flex-col gap-6 ">
      <div className="flex w-full items-center justify-between gap-4 space-y-0.5">
        <div className="flex w-full flex-col">
          <h2 className="flex items-center text-lg font-semibold tracking-tight">
            Shortcuts
            <ForwardedIconComponent
              name="Keyboard"
              className="ml-2 h-5 w-5 text-primary"
            />
          </h2>
          <p className="text-sm text-muted-foreground">
            Manage Shortcuts for quick access to frequently used actions.
          </p>
        </div>
        <div>
          <div className="align-end mb-4 flex w-full justify-end">
            <div className="justify center flex items-center">
              <EditShortcutButton
                disable={selectedRows.length === 0}
                defaultCombination={combinationToEdit[0]?.shortcut}
                shortcut={selectedRows}
                defaultShortcuts={shortcuts}
                open={open}
                setOpen={setOpen}
              >
                <Button variant="primary">
                  <ForwardedIconComponent name="Wrench" className="mr-2 w-4" />
                  Customize
                </Button>
              </EditShortcutButton>
              <Button
                variant="primary"
                className="ml-3"
                onClick={handleRestore}
              >
                <ForwardedIconComponent name="RotateCcw" className="mr-2 w-4" />
                Restore
              </Button>
            </div>
          </div>
        </div>
      </div>
      <div className="grid gap-6 pb-8">
        <div>
          <TableComponent
            onSelectionChanged={(event: SelectionChangedEvent) => {
              setSelectedRows(
                event.api.getSelectedRows().map((row) => row.name),
              );
            }}
            suppressRowClickSelection={true}
            domLayout="autoHeight"
            pagination={false}
            columnDefs={colDefs}
            rowData={nodesRowData}
            paginationPageSize={8}
          />
        </div>
      </div>
    </div>
  );
}
