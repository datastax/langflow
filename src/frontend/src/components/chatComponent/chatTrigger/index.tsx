import { Transition } from "@headlessui/react";
import {
  Bars3CenterLeftIcon,
  ChatBubbleBottomCenterTextIcon,
} from "@heroicons/react/24/outline";
import { MessagesSquare } from "lucide-react";
import { nodeColors } from "../../../utils";
import { alertContext } from "../../../contexts/alertContext";
import { useContext } from "react";
import ChatModal from "../../../modals/chatModal";

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
          className="border flex justify-center align-center py-1 px-3 w-12 h-12 rounded-full bg-amber-600 dark:border-gray-600 cursor-pointer"
          onClick={handleClick}
        >
          <button>
            <div className="flex gap-3  items-center">
              <MessagesSquare
                className="h-6 w-6 mt-1"
                style={{ color: "white" }}
              />
            </div>
          </button>
        </div>
      </div>
    </Transition>
  );
}
