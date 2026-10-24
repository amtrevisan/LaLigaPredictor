import { useEffect, useState } from "react";
import { useI18n } from "@/i18n";
import { decimal, fill, integer, kickoffDay, kickoffTime, longDate, percent, points } from "@/lib/format";
import type { Meeting, Team, Triple, UpcomingFixture } from "@/types/data";
import { Crest } from "./Crest";
import { DitherImage } from "./DitherImage";
import { ArrowUpRight } from "./icons";
import { OutcomeBar } from "./OutcomeBar";
import styles from "./FixtureDetail.module.css";

/**
 * One club photograph per fixture, stable across renders. TheSportsDB's first fan-art slot is
 * usually a crest illustration, so it is only used when it is the sole image.
 */
function pickPhoto(team: Team, seed: string): string | null {
  const pool = team.fanart.length > 1 ? team.fanart.slice(1) : team.fanart;
  if (pool.length === 0) return null;
  let hash = 0;
  for (const ch of seed) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return pool[hash % pool.length] ?? null;
}

function ProbabilityRow({ label, probs }: { label: string; probs: Triple }) {
  const { intl } = useI18n();
  return (
    <div className={styles.probRow}>
      <span className={styles.rowLabel}>{label}</span>
      <OutcomeBar probs={probs} orientation="horizontal" />
      <span className={styles.triple}>
        {probs.map((p, i) => (
          <span key={i}>{percent(p, intl)}</span>
        ))}
      </span>
    </div>
  );
}

/** Highlight links come from third-party data; only plain https URLs are rendered as links. */
function safeHttps(url: string | null): string | null {
  if (!url) return null;
  try {
    return new URL(url).protocol === "https:" ? url : null;
  } catch {
    return null;
  }
}

function MeetingTile({ meeting }: { meeting: Meeting }) {
  const { t, intl } = useI18n();
  const score = meeting.score ? `${meeting.score[0]}–${meeting.score[1]}` : "–";
  const caption = (
    <figcaption className={styles.caption}>
      <span className={styles.captionScore}>
        {meeting.home} {score} {meeting.away}
      </span>
      <span>
        {longDate(meeting.date, intl)}
        {meeting.competition ? ` · ${meeting.competition}` : ""}
      </span>
    </figcaption>
  );
  const placeholder = (
    <span className={styles.noImage} aria-hidden="true">
      {score}
    </span>
  );
  // The caption carries the facts, so the image itself is decorative.
  const image = meeting.image ? (
    <DitherImage src={meeting.image} alt="" fallback={placeholder} />
  ) : (
    placeholder
  );
  const video = safeHttps(meeting.video);
  return (
    <figure className={styles.tile}>
      {video ? (
        <a href={video} rel="noreferrer" target="_blank" className={styles.tileLink}>
          {image}
          <span className={styles.play}>
            {t.fixtures.highlights} <ArrowUpRight />
            <span className="visually-hidden">
              {" "}
              ({meeting.home} {score} {meeting.away})
            </span>
          </span>
        </a>
      ) : (
        image
      )}
      {caption}
    </figure>
  );
}

// The detail band opens in 1-bit and wipes to colour a beat later: the second half of the
// signature move that starts with the column inversion.
const REVEAL_DELAY_MS = 450;

export function FixtureDetail({ fixture, id }: { fixture: UpcomingFixture; id: string }) {
  const { t, intl } = useI18n();
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    // Remounted per selection (keyed by fixture), so this runs once per selected fixture.
    const timer = window.setTimeout(() => setRevealed(true), REVEAL_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, []);
  const gap = fixture.market ? fixture.probs.map((p, i) => p - (fixture.market?.[i] ?? 0)) : null;
  const topScore = fixture.scorelines[0]?.p ?? 1;
  // Club imagery either side of this match's own artwork, when each exists.
  const band = [
    ...[fixture.home].flatMap((team) => {
      const src = pickPhoto(team, fixture.id);
      return src ? [{ key: team.name, src, title: team.name, detail: null as string | null, club: true }] : [];
    }),
    ...(fixture.poster
      ? [
          {
            key: "poster",
            src: fixture.poster,
            title: t.fixtures.preview,
            detail: longDate(fixture.kickoff, intl),
            club: false,
          },
        ]
      : []),
    ...[fixture.away].flatMap((team) => {
      const src = pickPhoto(team, fixture.id);
      return src ? [{ key: team.name, src, title: team.name, detail: null as string | null, club: true }] : [];
    }),
  ];

  return (
    <section id={id} className={styles.detail} aria-labelledby={`${id}-title`}>
      <header className={styles.head}>
        <h2 id={`${id}-title`} className={styles.title}>
          <span className={styles.side}>
            <Crest team={fixture.home} size={56} />
            {fixture.home.name}
          </span>
          <span className={styles.versus} aria-hidden="true">
            –
          </span>
          <span className={`${styles.side} ${styles.away}`}>
            {fixture.away.name}
            <Crest team={fixture.away} size={56} />
          </span>
        </h2>
        <p className={styles.sub}>
          {kickoffDay(fixture.kickoff, intl)} · {kickoffTime(fixture.kickoff, intl)}
          {fixture.home.stadium ? ` · ${fixture.home.stadium}` : ""}
          {" · "}
          {fill(t.fixtures.round, { n: fixture.round })}
        </p>
      </header>

      {band.length > 0 && (
        <div className={styles.photos}>
          {band.map((item) => (
            <figure key={item.key} className={styles.photo}>
              <DitherImage
                src={item.src}
                alt=""
                // Only club colour belongs on the page: TheSportsDB's templated match artwork
                // stays 1-bit unless pointed at.
                revealed={revealed && item.club}
                fallback={<span className={styles.noImage} />}
              />
              <figcaption className={styles.caption}>
                <span className={styles.captionScore}>{item.title}</span>
                {item.detail && <span>{item.detail}</span>}
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      <div className={styles.grid}>
        <div className={styles.block}>
          <div className={styles.columnsHead} aria-hidden="true">
            <span />
            <span />
            <span className={styles.triple}>
              <span>{t.outcome.short.H}</span>
              <span>{t.outcome.short.D}</span>
              <span>{t.outcome.short.A}</span>
            </span>
          </div>
          <ProbabilityRow label={t.fixtures.model} probs={fixture.probs} />
          {fixture.market && <ProbabilityRow label={t.fixtures.market} probs={fixture.market} />}
          {gap && (
            <div className={styles.probRow}>
              <span className={styles.rowLabel}>{t.fixtures.gap}</span>
              <span />
              <span className={styles.triple}>
                {gap.map((g, i) => (
                  <span key={i}>{points(g, intl)}</span>
                ))}
              </span>
            </div>
          )}
          <dl className={styles.facts}>
            {fixture.components.classifier && (
              <div>
                <dt>{t.fixtures.classifier}</dt>
                <dd>{fixture.components.classifier.map((p) => percent(p, intl)).join(" · ")}</dd>
              </div>
            )}
            {fixture.components.goals && (
              <div>
                <dt>{t.fixtures.goals}</dt>
                <dd>{fixture.components.goals.map((p) => percent(p, intl)).join(" · ")}</dd>
              </div>
            )}
            {fixture.xg && (
              <div>
                <dt>{t.fixtures.xg}</dt>
                <dd>
                  {decimal(fixture.xg[0], intl, 2)} – {decimal(fixture.xg[1], intl, 2)}
                </dd>
              </div>
            )}
            {fixture.elo && (
              <div>
                <dt>{t.fixtures.elo}</dt>
                <dd>
                  {integer(fixture.elo[0], intl)} – {integer(fixture.elo[1], intl)}
                </dd>
              </div>
            )}
          </dl>
        </div>

        <div className={styles.block}>
          <h3 className={styles.blockTitle}>{t.fixtures.likeliest}</h3>
          <ol className={styles.scores}>
            {fixture.scorelines.map(({ score, p }) => (
              <li key={score.join("-")}>
                <span className={styles.score}>
                  {score[0]}–{score[1]}
                </span>
                <span className={styles.scoreBar} style={{ width: `${(p / topScore) * 100}%` }} />
                <span>{percent(p, intl, 1)}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <div className={styles.media}>
        <h3 className={styles.blockTitle}>{t.fixtures.meetings}</h3>
        <div className={styles.tiles}>
          {fixture.meetings.map((meeting) => (
            <MeetingTile key={`${meeting.date}-${meeting.home}`} meeting={meeting} />
          ))}
        </div>
        {fixture.meetings.length === 0 && <p className={styles.note}>{t.fixtures.noMeetings}</p>}
      </div>
    </section>
  );
}
