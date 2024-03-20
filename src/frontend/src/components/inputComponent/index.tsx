import * as Form from "@radix-ui/react-form";
import { PopoverAnchor } from "@radix-ui/react-popover";
import { useEffect, useRef, useState } from "react";
import { InputComponentType } from "../../types/components";
import { handleKeyDown } from "../../utils/reactflowUtils";
import { classNames, cn } from "../../utils/utils";
import ForwardedIconComponent from "../genericIconComponent";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../ui/command";
import { Input } from "../ui/input";
import { Popover, PopoverContentWithoutPortal } from "../ui/popover";

export default function InputComponent({
  autoFocus = false,
  onBlur,
  value = "",
  onChange,
  disabled,
  required = false,
  isForm = false,
  password,
  editNode = false,
  placeholder = "Type something...",
  className,
  id = "",
  blurOnEnter = false,
  optionsIcon = "ChevronsUpDown",
  selectedOption,
  setSelectedOption,
  options = [],
  optionsPlaceholder = "Search options...",
  optionsButton,
}: InputComponentType): JSX.Element {
  const [pwdVisible, setPwdVisible] = useState(false);
  const refInput = useRef<HTMLInputElement>(null);
  const [showOptions, setShowOptions] = useState<boolean>(false);

  // Clear component state
  useEffect(() => {
    if (disabled && value && onChange && value !== "") {
      onChange("");
    }
  }, [disabled]);

  function onInputLostFocus(event): void {
    if (onBlur) onBlur(event);
  }

  return (
    <div className="relative w-full">
      {isForm ? (
        <Form.Control asChild>
          <Input
            id={"form-" + id}
            ref={refInput}
            onBlur={onInputLostFocus}
            autoFocus={autoFocus}
            type={password && !pwdVisible ? "password" : "text"}
            value={value}
            disabled={disabled}
            required={required}
            className={classNames(
              password && !pwdVisible && value !== ""
                ? " text-clip password "
                : "",
              editNode ? " input-edit-node " : "",
              password && editNode ? "pr-8" : "",
              password && !editNode ? "pr-10" : "",
              className!
            )}
            placeholder={password && editNode ? "Key" : placeholder}
            onChange={(e) => {
              onChange && onChange(e.target.value);
            }}
            onCopy={(e) => {
              e.preventDefault();
            }}
            onKeyDown={(e) => {
              handleKeyDown(e, value, "");
              if (blurOnEnter && e.key === "Enter") refInput.current?.blur();
            }}
          />
        </Form.Control>
      ) : (
        <>
          <Popover open={showOptions} onOpenChange={setShowOptions}>
            <PopoverAnchor>
              <Input
                id={id}
                ref={refInput}
                type="text"
                onBlur={onInputLostFocus}
                value={
                  selectedOption !== "" || !onChange ? selectedOption : value
                }
                autoFocus={autoFocus}
                disabled={disabled}
                onClick={() => {
                  (selectedOption !== "" || !onChange) && setShowOptions(true);
                }}
                required={required}
                className={classNames(
                  password &&
                    selectedOption === "" &&
                    !pwdVisible &&
                    value !== ""
                    ? " text-clip password "
                    : "",
                  editNode ? " input-edit-node " : "",
                  password && selectedOption === "" && editNode ? "pr-8" : "",
                  password && selectedOption === "" && !editNode ? "pr-10" : "",
                  className!
                )}
                placeholder={password && editNode ? "Key" : placeholder}
                onChange={(e) => {
                  onChange && onChange(e.target.value);
                }}
                onKeyDown={(e) => {
                  handleKeyDown(e, value, "");
                  if (blurOnEnter && e.key === "Enter")
                    refInput.current?.blur();
                }}
                data-testid={editNode ? id + "-edit" : id}
              />
            </PopoverAnchor>
            <PopoverContentWithoutPortal
              className="nocopy nopan nodelete nodrag noundo p-0"
              style={{ minWidth: refInput?.current?.clientWidth ?? "200px" }}
              side="bottom"
              align="center"
            >
              <Command>
                <CommandInput placeholder={optionsPlaceholder} />
                <CommandList>
                  <CommandEmpty>No results found.</CommandEmpty>
                  <CommandGroup defaultChecked={false}>
                    {options.map((option, id) => (
                      <CommandItem
                        key={id}
                        value={option}
                        onSelect={(currentValue) => {
                          setSelectedOption &&
                            setSelectedOption(
                              currentValue === selectedOption
                                ? ""
                                : currentValue
                            );
                          setShowOptions(false);
                        }}
                      >
                        <ForwardedIconComponent
                          name="Check"
                          className={cn(
                            "mr-2 h-4 w-4 text-primary",
                            selectedOption === option
                              ? "opacity-100"
                              : "opacity-0"
                          )}
                          aria-hidden="true"
                        />
                        {option}
                      </CommandItem>
                    ))}
                    {optionsButton && optionsButton}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContentWithoutPortal>
          </Popover>
          <div
            className={cn(
              "pointer-events-auto absolute inset-y-0 h-full w-full cursor-pointer",
              selectedOption !== "" || !onChange ? "" : "hidden"
            )}
            onClick={
              selectedOption !== "" || !onChange
                ? (e) => {
                    setShowOptions((old) => !old);
                    e.preventDefault();
                    e.stopPropagation();
                  }
                : () => {}
            }
          ></div>
        </>
      )}

      <span
        className={cn(
          password && selectedOption === "" ? "right-8" : "right-0",
          "absolute inset-y-0 flex items-center pr-2.5"
        )}
      >
        <button
          onClick={() => {
            setShowOptions(!showOptions);
          }}
          className={cn(
            selectedOption !== ""
              ? "text-medium-indigo"
              : "text-muted-foreground",
            "hover:text-accent-foreground"
          )}
        >
          <ForwardedIconComponent
            name={optionsIcon}
            className={"h-4 w-4"}
            aria-hidden="true"
          />
        </button>
      </span>

      {password && selectedOption === "" && (
        <button
          type="button"
          tabIndex={-1}
          className={classNames(
            "mb-px",
            editNode
              ? "input-component-true-button"
              : "input-component-false-button"
          )}
          onClick={(event) => {
            event.preventDefault();
            setPwdVisible(!pwdVisible);
          }}
        >
          {password &&
            selectedOption === "" &&
            (pwdVisible ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className={classNames(
                  editNode
                    ? "input-component-true-svg"
                    : "input-component-false-svg"
                )}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className={classNames(
                  editNode
                    ? "input-component-true-svg"
                    : "input-component-false-svg"
                )}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            ))}
        </button>
      )}
    </div>
  );
}
