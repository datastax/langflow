import { useNavigate } from "react-router-dom";
import useFlowsManagerStore from "../../../../stores/flowsManagerStore";
import NewFlowModal from "../../../../modals/newFlowModal";
import { useState } from "react";

type EmptyComponentProps = {};

const EmptyComponent = ({}: EmptyComponentProps) => {
  const addFlow = useFlowsManagerStore((state) => state.addFlow);
  const navigate = useNavigate();

  const [openModal, setOpenModal] = useState(false);

  return (
    <>
      <div className="mt-6 flex w-full items-center justify-center text-center">
        <div className="flex-max-width h-full flex-col">
          <div className="align-center flex w-full justify-center gap-1 ">
            <span className="text-muted-foreground">
              This folder is empty. New?
            </span>
            <span className="transition-colors hover:text-muted-foreground">
              <NewFlowModal open={openModal} setOpen={setOpenModal} />
              <button
                onClick={() => {
                  setOpenModal(true);
                }}
                className="underline"
              >
                Start Here
              </button>
            </span>
            <span className="animate-pulse">🚀</span>
          </div>
        </div>
      </div>
    </>
  );
};
export default EmptyComponent;
