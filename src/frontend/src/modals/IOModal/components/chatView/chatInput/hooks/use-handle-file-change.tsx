import ShortUniqueId from "short-unique-id";
import useFileUpload from "./use-file-upload";
import useAlertStore from "../../../../../../stores/alertStore";

const fsErrorText =
  "Please ensure your file has one of the following extensions:";
const snErrorTxt = "png, jpg, jpeg, gif, bmp, webp";

export const useHandleFileChange = (setFiles, currentFlowId) => {
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const fileInput = event.target;
    const file = fileInput.files?.[0];
    if (file) {
      const allowedExtensions = ["png", "jpg", "jpeg", "gif", "bmp", "webp"];
      const fileExtension = file.name.split(".").pop()?.toLowerCase();

      if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
        setErrorData({
          title: "Error uploading file",
          list: [fsErrorText, snErrorTxt],
        });
        return;
      }

      const uid = new ShortUniqueId({ length: 10 }); // Increase the length to ensure uniqueness
      const id = uid();
      const type = file.type.split("/")[0];
      const blob = file;

      setFiles((prevFiles) => [
        ...prevFiles,
        { file: blob, loading: true, error: false, id, type },
      ]);

      useFileUpload(blob, currentFlowId, setFiles, id);
    }

    // Clear the file input value to ensure the change event is triggered even for the same file
    fileInput.value = "";
  };

  return {
    handleFileChange,
  };
};

export default useHandleFileChange;
