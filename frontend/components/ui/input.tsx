import * as React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-1 text-sm text-white shadow-sm transition-colors",
          "placeholder:text-slate-500",
          "focus-visible:outline-none focus-visible:border-brand-500 focus-visible:ring-1 focus-visible:ring-brand-500/40",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
