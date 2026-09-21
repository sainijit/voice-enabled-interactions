/**
 * VoiceToVoiceTable — "Voice-to-Voice per Turn" history table.
 *
 * Surfaces the exact metrics tracked while closing the ASR-latency gap
 * against kiosk-voice-lab-main (~380ms -> ~120-180ms):
 *
 *   - ASR latency        (asr.last_word_to_transcript_ms) — target ~180-220ms
 *   - Post-speech gap    (wall.post_speech_gap_ms)         — mic-release reaction time
 *   - Final flush wait   (wall.final_flush_wait_ms)        — last commit round-trip
 *   - Endpoint wait      (wall.endpoint_wait_ms)            — deliberate silence pause
 *   - LLM TTFT           (agent.ttft_ms)                    — agent-start to first reply token
 *   - Processing latency (wall.voice_to_voice_post_endpoint_ms) — turn-end decision to
 *                                                                 first sound (shared cross-team
 *                                                                 vocabulary term; see
 *                                                                 benchmark-vocabolary.txt).
 *                                                                 NOT time_to_first_audio_ms —
 *                                                                 the two differ whenever work
 *                                                                 starts speculatively during
 *                                                                 the endpoint wait.
 *   - V2V (customer)     (wall.voice_to_voice_ms)           — full customer-felt clock
 *
 * Source: kiosk-core /api/v1/pipeline/recent (ring buffer of the last 20
 * completed turns), already fetched into kpis.pipelineRecent by useKpis —
 * this component is the first consumer of that field.
 */

import type { PipelineTurnTrace } from '../../types';

interface VoiceToVoiceTableProps {
  turns: PipelineTurnTrace[];
}

const ms = (v: number | null | undefined): string =>
  typeof v === 'number' ? `${Math.round(v)}` : '—';

// ASR-latency target from the lab comparison: green under 220ms, amber
// under 380ms (the old baseline), red at/above the old baseline.
function asrCellClass(v: number | null | undefined): string {
  if (typeof v !== 'number') return 'text-gray-400';
  if (v <= 220) return 'text-green-600 font-semibold';
  if (v < 380) return 'text-amber-600 font-semibold';
  return 'text-red-600 font-semibold';
}

function shortcutBadge(fired: boolean | null | undefined): JSX.Element {
  if (fired === true) {
    return (
      <span className="rounded bg-green-50 px-1.5 py-0.5 text-[10px] font-semibold text-green-700">
        shortcut
      </span>
    );
  }
  if (fired === false) {
    return (
      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold text-gray-500">
        full wait
      </span>
    );
  }
  return <span className="text-gray-300">—</span>;
}

// Simple median over whatever non-null numeric samples are present — used
// for the headline summary strip above the table (small sample sizes here,
// so median is more demo-stable than mean against the odd outlier turn).
function median(values: Array<number | null | undefined>): number | null {
  const nums = values.filter((v): v is number => typeof v === 'number').sort((a, b) => a - b);
  if (nums.length === 0) return null;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 === 0 ? (nums[mid - 1] + nums[mid]) / 2 : nums[mid];
}

function SummaryStat({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="flex flex-col items-center px-3 py-1.5">
      <span className={`font-mono text-base font-bold ${className ?? 'text-gray-700'}`}>{value}</span>
      <span className="text-[10px] uppercase tracking-wide text-gray-400">{label}</span>
    </div>
  );
}

// Stacked mini-bar visualising how voice_to_voice_ms splits across its
// additive components: endpoint_wait_ms + post_speech_gap_ms +
// final_flush_wait_ms + time_to_first_audio_ms == voice_to_voice_ms, with
// time_to_first_audio_ms itself further split into LLM TTFT (agent.ttft_ms)
// and the remaining TTS-compute time, so the demo can visually point at
// "here's where the time goes" instead of just reading raw numbers off
// separate columns.
const BREAKDOWN_SEGMENTS: Array<{ key: 'endpoint' | 'gap' | 'flush' | 'llm' | 'tts'; color: string; label: string }> = [
  { key: 'endpoint', color: 'bg-gray-400', label: 'Endpoint wait (deliberate silence pause)' },
  { key: 'gap', color: 'bg-indigo-400', label: 'Post-speech gap (mic-release reaction time)' },
  { key: 'flush', color: 'bg-amber-400', label: 'Final flush (last-chunk ASR round-trip)' },
  { key: 'llm', color: 'bg-purple-500', label: 'LLM TTFT (agent-start to first reply token)' },
  { key: 'tts', color: 'bg-intel-blue', label: 'TTS compute (first segment synth + WAV write)' },
];

function V2vBreakdownBar({ turn }: { turn: PipelineTurnTrace }) {
  const ttfa = turn.wall.time_to_first_audio_ms ?? 0;
  const llmTtft = Math.min(turn.agent.ttft_ms ?? 0, ttfa);
  const parts = {
    endpoint: turn.wall.endpoint_wait_ms ?? 0,
    gap: Math.max(turn.wall.post_speech_gap_ms ?? 0, 0),
    flush: turn.wall.final_flush_wait_ms ?? 0,
    llm: llmTtft,
    tts: Math.max(ttfa - llmTtft, 0),
  };
  const total = parts.endpoint + parts.gap + parts.flush + parts.llm + parts.tts;
  if (total <= 0) return <span className="text-gray-300">—</span>;
  return (
    <div className="flex h-3 w-28 overflow-hidden rounded-sm bg-gray-100" title={
      BREAKDOWN_SEGMENTS.map((s) => `${s.label}: ${ms(parts[s.key])}ms`).join('\n')
    }>
      {BREAKDOWN_SEGMENTS.map((s) => {
        const pct = (parts[s.key] / total) * 100;
        if (pct <= 0) return null;
        return <div key={s.key} className={s.color} style={{ width: `${pct}%` }} />;
      })}
    </div>
  );
}

export function VoiceToVoiceTable({ turns }: VoiceToVoiceTableProps) {
  // Newest first for the table — pipelineRecent arrives oldest-first.
  const rows = [...turns].reverse();

  const medianProcessing = median(rows.map((t) => t.wall.voice_to_voice_post_endpoint_ms));
  const medianCustomer = median(rows.map((t) => t.wall.voice_to_voice_ms));
  const medianAsr = median(rows.map((t) => t.asr.last_word_to_transcript_ms));
  const medianTtft = median(rows.map((t) => t.agent.ttft_ms));

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 bg-gray-50 px-3 py-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
          🔊 Voice-to-Voice per Turn
        </span>
      </div>

      {rows.length > 0 && (
        <div className="flex flex-wrap items-center justify-center gap-1 divide-x divide-gray-100 border-b border-gray-100 bg-blue-50/30 py-1">
          <SummaryStat label="Processing latency (median)" value={`${ms(medianProcessing)}ms`} className="text-intel-blue" />
          <SummaryStat label="V2V customer (median)" value={`${ms(medianCustomer)}ms`} className="text-gray-700" />
          <SummaryStat label="ASR latency (median)" value={`${ms(medianAsr)}ms`} className={asrCellClass(medianAsr)} />
          <SummaryStat label="LLM TTFT (median)" value={`${ms(medianTtft)}ms`} className="text-purple-600" />
          <span className="px-3 py-1.5 text-[10px] text-gray-400">n={rows.length} turns</span>
        </div>
      )}

      {rows.length === 0 ? (
        <p className="px-3 py-4 text-center text-xs text-gray-400">
          No completed turns yet — speak to the assistant to populate this table.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-gray-100 text-gray-400">
                <th className="px-2 py-1.5 text-left font-medium">Turn</th>
                <th className="px-2 py-1.5 text-right font-medium">ASR latency</th>
                <th className="px-2 py-1.5 text-right font-medium">Post-speech gap</th>
                <th className="px-2 py-1.5 text-right font-medium">Final flush</th>
                <th className="px-2 py-1.5 text-right font-medium">Endpoint wait</th>
                <th className="px-2 py-1.5 text-right font-medium">LLM TTFT</th>
                <th className="px-2 py-1.5 text-right font-medium" title="Turn-end decision to first sound (shared vocabulary: processing latency)">Processing</th>
                <th className="px-2 py-1.5 text-right font-medium">V2V (customer)</th>
                <th className="px-2 py-1.5 text-left font-medium">Breakdown</th>
                <th className="px-2 py-1.5 text-center font-medium">Endpoint</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t, i) => (
                <tr
                  key={t.turn_id}
                  className={`border-b border-gray-50 last:border-0 ${i === 0 ? 'bg-blue-50/40' : ''}`}
                >
                  <td className="px-2 py-1.5 font-mono text-gray-500">
                    {rows.length - i}
                  </td>
                  <td className={`px-2 py-1.5 text-right font-mono ${asrCellClass(t.asr.last_word_to_transcript_ms)}`}>
                    {ms(t.asr.last_word_to_transcript_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-gray-500">
                    {ms(t.wall.post_speech_gap_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-gray-600">
                    {ms(t.wall.final_flush_wait_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-gray-600">
                    {ms(t.wall.endpoint_wait_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-purple-600">
                    {ms(t.agent.ttft_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono font-semibold text-intel-blue">
                    {ms(t.wall.voice_to_voice_post_endpoint_ms)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-gray-600">
                    {ms(t.wall.voice_to_voice_ms)}
                  </td>
                  <td className="px-2 py-1.5">
                    <V2vBreakdownBar turn={t} />
                  </td>
                  <td className="px-2 py-1.5 text-center">
                    {shortcutBadge(t.wall.endpoint_shortcut_fired)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="border-t border-gray-100 px-3 py-1.5 text-[10px] leading-snug text-gray-400">
        ASR latency: <span className="text-green-600 font-semibold">green</span> ≤220ms target,{' '}
        <span className="text-amber-600 font-semibold">amber</span> improving,{' '}
        <span className="text-red-600 font-semibold">red</span> ≥380ms original baseline. All values in ms.
      </p>
      <p className="border-t border-gray-100 px-3 py-1.5 text-[10px] leading-snug text-gray-400">
        Breakdown bar (hover a bar for exact ms):{' '}
        <span className="inline-block h-2 w-2 rounded-sm bg-gray-400 align-middle" /> endpoint wait{' '}
        <span className="inline-block h-2 w-2 rounded-sm bg-indigo-400 align-middle" /> post-speech gap{' '}
        <span className="inline-block h-2 w-2 rounded-sm bg-amber-400 align-middle" /> final flush{' '}
        <span className="inline-block h-2 w-2 rounded-sm bg-purple-500 align-middle" /> LLM TTFT{' '}
        <span className="inline-block h-2 w-2 rounded-sm bg-intel-blue align-middle" /> TTS compute.
        These five segments sum exactly to V2V (customer).
      </p>
    </div>
  );
}

export default VoiceToVoiceTable;
