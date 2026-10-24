import { useI18n } from "@/i18n";
import { percent } from "@/lib/format";
import type { CalibrationBin } from "@/types/data";
import styles from "./CalibrationChart.module.css";

const SIZE = 320;
const PAD = 36;
const SPAN = SIZE - PAD * 2;
const TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

const x = (p: number) => PAD + p * SPAN;
const y = (p: number) => SIZE - PAD - p * SPAN;

interface Props {
  model: CalibrationBin[];
  market: CalibrationBin[];
}

/** Reliability diagram: model bins as solid squares, market bins as outlines, area ∝ count. */
export function CalibrationChart({ model, market }: Props) {
  const { t, intl } = useI18n();
  const maxCount = Math.max(...model.map((b) => b.count), ...market.map((b) => b.count), 1);
  const side = (count: number) => 4 + 10 * Math.sqrt(count / maxCount);
  const describe = (bins: CalibrationBin[]) =>
    bins.map((b) => `${percent(b.predicted, intl)} → ${percent(b.observed, intl)}`).join(", ");

  return (
    <figure className={styles.figure}>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className={styles.svg}
        role="img"
        aria-labelledby="calibration-title calibration-desc"
      >
        <title id="calibration-title">{t.model.calibration}</title>
        <desc id="calibration-desc">
          {`${t.model.blend}: ${describe(model)}. ${t.model.market}: ${describe(market)}.`}
        </desc>
        {TICKS.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={y(0)} y2={y(1)} className={styles.grid} />
            <line x1={x(0)} x2={x(1)} y1={y(tick)} y2={y(tick)} className={styles.grid} />
            <text x={x(tick)} y={y(0) + 16} className={styles.tick} textAnchor="middle">
              {Math.round(tick * 100)}
            </text>
            <text x={x(0) - 8} y={y(tick) + 4} className={styles.tick} textAnchor="end">
              {Math.round(tick * 100)}
            </text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className={styles.diagonal} />
        {market.map((b, i) => {
          const s = side(b.count);
          return (
            <rect
              key={`market-${i}`}
              x={x(b.predicted) - s / 2}
              y={y(b.observed) - s / 2}
              width={s}
              height={s}
              className={styles.market}
            />
          );
        })}
        {model.map((b, i) => {
          const s = side(b.count);
          return (
            <rect
              key={`model-${i}`}
              x={x(b.predicted) - s / 2}
              y={y(b.observed) - s / 2}
              width={s}
              height={s}
              className={styles.model}
            />
          );
        })}
      </svg>
      <figcaption className={styles.caption}>
        <span className={styles.axis}>
          {t.model.predicted} (x) · {t.model.observed} (y), %
        </span>
        <span className={styles.legend}>
          <span className={styles.swatchModel} aria-hidden="true" /> {t.model.blend}
          <span className={styles.swatchMarket} aria-hidden="true" /> {t.model.market}
        </span>
      </figcaption>
    </figure>
  );
}
