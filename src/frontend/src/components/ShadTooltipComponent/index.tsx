import { ShadToolTipType } from "../../types/components";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

export default function ShadTooltip({
  content,
  side,
  asChild = true,
  children,
  style,
  delayDuration = 500,
}: ShadToolTipType) {
  return (
    <Tooltip delayDuration={delayDuration}>
      <TooltipTrigger asChild={asChild}>{children}</TooltipTrigger>

      <TooltipContent
        className={style}
        side={side}
        avoidCollisions={false}
        sticky="always"
      >
        {content}
      </TooltipContent>
    </Tooltip>
  );
}
