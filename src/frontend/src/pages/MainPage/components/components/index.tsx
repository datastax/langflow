import { useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import PaginatorComponent from "../../../../components/PaginatorComponent";
import CollectionCardComponent from "../../../../components/cardComponent";
import CardsWrapComponent from "../../../../components/cardsWrapComponent";
import IconComponent from "../../../../components/genericIconComponent";
import { SkeletonCardComponent } from "../../../../components/skeletonCardComponent";
import { Button } from "../../../../components/ui/button";
import { alertContext } from "../../../../contexts/alertContext";
import { FlowsContext } from "../../../../contexts/flowsContext";
import { FlowType } from "../../../../types/flow";

export default function ComponentsComponent({
  is_component = true,
}: {
  is_component?: boolean;
}) {
  const { flows, removeFlow, uploadFlow, isLoading } = useContext(FlowsContext);
  const { setErrorData, setSuccessData } = useContext(alertContext);
  const [pageSize, setPageSize] = useState(10);
  const [pageIndex, setPageIndex] = useState(1);
  const [allData, setAllData] = useState(flows);

  const navigate = useNavigate();

  useEffect(() => {
    setAllData(
      flows
        .filter((f) => f.is_component === is_component)
        .sort((a, b) => {
          if (a?.updated_at && b?.updated_at) {
            return (
              new Date(b?.updated_at!).getTime() -
              new Date(a?.updated_at!).getTime()
            );
          } else if (a?.updated_at && !b?.updated_at) {
            return -1;
          } else if (!a?.updated_at && b?.updated_at) {
            return 1;
          } else {
            return (
              new Date(b?.date_created!).getTime() -
              new Date(a?.date_created!).getTime()
            );
          }
        })
    );
  }, [flows]);

  useEffect(() => {
    const start = (pageIndex - 1) * pageSize;
    const end = start + pageSize;
    setData(allData.slice(start, end));
  }, [pageIndex, pageSize, allData]);

  const [data, setData] = useState<FlowType[]>([]);

  const name = is_component ? "Component" : "Flow";

  const onFileDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.types.some((types) => types === "Files")) {
      if (e.dataTransfer.files.item(0).type === "application/json") {
        uploadFlow({
          newProject: true,
          file: e.dataTransfer.files.item(0)!,
          isComponent: is_component,
        })
          .then(() => {
            setSuccessData({
              title: `${
                is_component ? "Component" : "Flow"
              } uploaded successfully`,
            });
          })
          .catch((error) => {
            setErrorData({
              title: "Error uploading file",
              list: [error],
            });
          });
      } else {
        setErrorData({
          title: "Invalid file type",
          list: ["Please upload a JSON file"],
        });
      }
    }
  };

  return (
    <CardsWrapComponent
      onFileDrop={onFileDrop}
      dragMessage={`Drag your ${name} here`}
    >
      <div className="flex h-full w-full flex-col justify-between">
        <div className="flex w-full flex-col gap-4">
          <div className="grid w-full gap-4 md:grid-cols-2 lg:grid-cols-2">
            {!isLoading || data?.length > 0 ? (
              data?.map((item, idx) => (
                <CollectionCardComponent
                  onDelete={() => {
                    removeFlow(item.id);
                  }}
                  key={idx}
                  data={item}
                  disabled={isLoading}
                  button={
                    !is_component ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="whitespace-nowrap "
                        onClick={() => {
                          navigate("/flow/" + item.id);
                        }}
                      >
                        <IconComponent
                          name="ExternalLink"
                          className="main-page-nav-button"
                        />
                        Edit Flow
                      </Button>
                    ) : (
                      <></>
                    )
                  }
                />
              ))
            ) : !isLoading && data?.length === 0 ? (
              <>You haven't created {name}s yet.</>
            ) : (
              <>
                <SkeletonCardComponent />
                <SkeletonCardComponent />
              </>
            )}
          </div>
        </div>
        {!isLoading && allData.length > 0 && (
          <div className="relative py-6">
            <PaginatorComponent
              storeComponent={true}
              pageIndex={pageIndex}
              pageSize={pageSize}
              rowsCount={[10, 20, 50, 100]}
              totalRowsCount={allData.length}
              paginate={(pageSize, pageIndex) => {
                setPageIndex(pageIndex);
                setPageSize(pageSize);
              }}
            ></PaginatorComponent>
          </div>
        )}
      </div>
    </CardsWrapComponent>
  );
}
