import { useEffect, useState } from "react";
import { FloatComponentType } from "../../types/components";
import { Input } from "../ui/input";

export default function FloatComponent({
  value,
  onChange,
  disabled,
  editNode = false,
}: FloatComponentType) {
  const step = 0.1;
  const min = 0;
  const max = 1;

  const [myValue, setMyValue] = useState(value);

  useEffect(() => {
    setMyValue(value);
  }, [value]);

  // Clear component state
  useEffect(() => {
    if (disabled) {
      onChange("");
    }
  }, [disabled, onChange]);

  return (
    <div className="w-full">
      <Input
        type="number"
        step={step}
        min={min}
        onInput={(e: React.ChangeEvent<HTMLInputElement>) => {
          if (e.target.value < min.toString()) {
            e.target.value = min.toString();
          }
          if (e.target.value > max.toString()) {
            e.target.value = max.toString();
          }
        }}
        max={max}
        value={value ?? ""}
        disabled={disabled}
        className={editNode ? "input-edit-node" : ""}
        placeholder={
          editNode ? "Number 0 to 1" : "Type a number from zero to one"
        }
        onChange={(e) => {
          onChange(e.target.value);
          setMyValue(e.target.value);
        }}
      />
    </div>
  );
}
