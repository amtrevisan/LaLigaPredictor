// Hairline icons drawn on a 12px grid, 1.25px stroke, square caps. Decorative by default.

interface IconProps {
  size?: number;
}

export function ArrowUpRight({ size = 12 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M3 9 9 3M4 3h5v5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="square" />
    </svg>
  );
}

export function ArrowRight({ size = 12 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M1.5 6h8.5M7 3l3 3-3 3" stroke="currentColor" strokeWidth="1.25" strokeLinecap="square" />
    </svg>
  );
}
