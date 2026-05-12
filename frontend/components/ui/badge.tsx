import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-brand-600/40 bg-brand-600/15 text-brand-400",
        success: "border-green-600/40 bg-green-600/15 text-green-400",
        warning: "border-yellow-600/40 bg-yellow-600/15 text-yellow-400",
        destructive: "border-red-600/40 bg-red-600/15 text-red-400",
        secondary: "border-white/10 bg-white/5 text-slate-400",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
