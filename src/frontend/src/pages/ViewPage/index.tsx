import { useContext, useEffect } from "react";
import { useParams } from "react-router-dom";
import Page from "../FlowPage/components/PageComponent";

export default function ViewPage() {
  const { setDark } = useContext(darkContext);
  const { id, theme } = useParams();

  // Set flow tab id
  useEffect(() => {
    setTabId(id!);
  }, [id]);

  return (
    <div className="flow-page-positioning">
      {flows.length > 0 &&
        tabId !== "" &&
        flows.findIndex((flow) => flow.id === tabId) !== -1 && (
          <Page view flow={flows.find((flow) => flow.id === tabId)!} />
        )}
    </div>
  );
}
