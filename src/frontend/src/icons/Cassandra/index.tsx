import React, { forwardRef } from "react";
import CassandraSVG from "./Cassandra";

export const CassandraIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <CassandraSVG ref={ref} {...props} />;
});
