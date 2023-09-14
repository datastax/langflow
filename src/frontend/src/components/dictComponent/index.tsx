import { useEffect } from "react";
import { DictComponentType } from "../../types/components";

import DictAreaModal from "../../modals/dictAreaModal";
import { classNames } from "../../utils/utils";
import { Input } from "../ui/input";

export default function DictComponent({
  value,
  onChange,
  disabled,
  editNode = false,
}: DictComponentType): JSX.Element {
  useEffect(() => {
    if (disabled) {
      onChange([""]);
    }
  }, [disabled]);

  return (
    <div
      className={classNames(
        value.length > 1 && editNode ? "my-1" : "",
        "flex flex-col gap-3"
      )}
    >
      {
        <div className="flex w-full gap-3">
          <DictAreaModal
            value={value}
            onChange={(obj) => {
              onChange(obj);
            }}
          >
            <Input
              disabled={true}
              type="text"
              className={
                editNode ? "input-edit-node cursor-pointer" : "cursor-pointer"
              }
              placeholder="Click to edit your dictionary..."
            />
          </DictAreaModal>
        </div>
      }
    </div>
  );
}
