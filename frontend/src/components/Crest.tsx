import { useState } from "react";
import type { Team } from "@/types/data";
import styles from "./Crest.module.css";

interface Props {
  team: Team;
  size?: number;
}

/** Club crest, the only colour on the page. Falls back to the club's short code. */
export function Crest({ team, size = 28 }: Props) {
  const [failed, setFailed] = useState(false);
  if (!team.crest || failed) {
    return (
      <span className={styles.fallback} style={{ width: size, height: size }} aria-hidden="true">
        {team.short.slice(0, 3)}
      </span>
    );
  }
  return (
    <img
      className={styles.crest}
      src={team.crest}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}
