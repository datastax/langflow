import { ArrowTopRightOnSquareIcon } from "@heroicons/react/24/outline";
import { useContext, useEffect, useState } from "react";
import { PopUpContext } from "../../contexts/popUpContext";
import { TextAreaComponentType } from "../../types/components";
import GenericModal from "../../modals/genericModal";
import { TypeModal } from "../../utils";

export default function TextAreaComponent({
  value,
  onChange,
  disabled,
  editNode = false,
}: TextAreaComponentType) {
  const [myValue, setMyValue] = useState(value);
  const { openPopUp, closePopUp } = useContext(PopUpContext);

  useEffect(() => {
    if (disabled) {
      setMyValue("");
      onChange("");
    }
  }, [disabled, onChange]);

  useEffect(() => {
    setMyValue(value);
  }, [closePopUp]);

  return (
    <div className={disabled ? "pointer-events-none cursor-not-allowed" : ""}>
      <div
        className={
          editNode
            ? "w-full flex items-center"
            : "w-full flex items-center gap-3"
        }
      >
        <span
          onClick={() => {
            openPopUp(
              <GenericModal
                type={TypeModal.TEXT}
                buttonText="Finishing Editing"
                modalTitle="Edit Text"
                value={myValue}
                setValue={(t: string) => {
                  setMyValue(t);
                  onChange(t);
                }}
              />
            );
          }}
          className={
            editNode
              ? "truncate cursor-pointer placeholder:text-center text-gray-500 border-1 block w-full pt-0.5 pb-0.5 form-input dark:bg-gray-900 dark:text-gray-300 dark:border-gray-600 rounded-md border-gray-300 shadow-sm sm:text-sm focus:ring-1 focus:ring-offset-1 focus:ring-ring"
              : "truncate block w-full text-gray-500 dark:text-muted px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm" +
                (disabled ? " bg-gray-200" : "")
          }
        >
          {myValue !== "" ? myValue : "Type something..."}
        </span>
        <button
          onClick={() => {
            openPopUp(
              <GenericModal
                type={TypeModal.TEXT}
                buttonText="Finishing Editing"
                modalTitle="Edit Text"
                value={myValue}
                setValue={(t: string) => {
                  setMyValue(t);
                  onChange(t);
                }}
              />
            );
          }}
        >
          {!editNode && (
            <ArrowTopRightOnSquareIcon className="w-6 h-6 hover:text-blue-600 dark:text-gray-300" />
          )}
        </button>
      </div>
    </div>
  );
}
