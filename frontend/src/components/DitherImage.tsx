import { useState } from "react";
import styles from "./DitherImage.module.css";

interface Props {
  src: string;
  alt: string;
  /** Force the colour state (used when the surrounding fixture is selected). */
  revealed?: boolean;
  className?: string;
}

/**
 * A photograph held as a 1-bit ordered dither until pointed at or focused, when it resolves to
 * colour. The colour image sits underneath; the dithered layer fades out over it.
 *
 * The dither is pure CSS (grayscale, a Bayer tile overlay-blended on top, then a hard contrast
 * on the group), so it works on hot-linked cross-origin images: no canvas, no CORS, and no SVG
 * `feImage`, which Chrome refuses to render inside filters.
 */
export function DitherImage({ src, alt, revealed = false, className }: Props) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <span className={[styles.frame, revealed && styles.revealed, className].filter(Boolean).join(" ")}>
      <img src={src} alt={alt} loading="lazy" decoding="async" onError={() => setFailed(true)} />
      <span className={styles.dither} aria-hidden="true">
        <img src={src} alt="" loading="lazy" decoding="async" />
        <span className={styles.bayer} />
      </span>
    </span>
  );
}
