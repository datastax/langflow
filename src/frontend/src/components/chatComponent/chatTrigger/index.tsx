import { Transition } from "@headlessui/react";
import { MessagesSquare } from "lucide-react";
import { alertContext } from "../../../contexts/alertContext";
import { useContext } from "react";

export default function ChatTrigger({ open, setOpen, isBuilt }) {
  const { setErrorData } = useContext(alertContext);

  function handleClick() {
    if (isBuilt) {
      setOpen(true);
    } else {
      setErrorData({
        title: "Flow not built",
        list: ["Please build the flow before chatting"],
      });
    }
  }

  return (
    <Transition
      show={!open}
      appear={true}
      enter="transition ease-out duration-300"
      enterFrom="translate-y-96"
      enterTo="translate-y-0"
      leave="transition ease-in duration-300"
      leaveFrom="translate-y-0"
      leaveTo="translate-y-96"
    >
      <div className="absolute bottom-4 right-3">
        <div
          className="border flex justify-center items-center py-1 px-3 w-12 h-12 rounded-full bg-blue-500 dark:border-gray-600 cursor-pointer"
          onClick={handleClick}
        >
          <button>
            <div className="flex gap-3">
              <MessagesSquare
                className="h-6 w-6  text-blue-100 fill-blue-100"
                style={{ color: "white" }}
                strokeWidth={1.5}
              />
            </div>
          </button>
        </div>
      </div>
    </Transition>
  );
}
