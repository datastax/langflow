import React, { forwardRef } from "react";
import SvgReadthedocsioIcon from "./ReadthedocsioIcon";

export const ReadTheDocsIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgReadthedocsioIcon ref={ref} {...props} />;
});
