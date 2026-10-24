// The Oddsless mark: one outcome column laid on its side, home solid, draw barcode, away outline,
// stepping down like a prediction. Same glyph as the favicon; drawn in currentColor.

export function LogoMark({ size = 18 }: { size?: number }) {
  return (
    <svg width={(size * 20) / 16} height={size} viewBox="0 0 20 16" fill="none" aria-hidden="true">
      <rect width="6" height="16" fill="currentColor" />
      <path d="M8.5 6v10M10.5 6v10M12.5 6v10" stroke="currentColor" />
      <rect x="14.5" y="9.5" width="5" height="6" stroke="currentColor" />
    </svg>
  );
}
