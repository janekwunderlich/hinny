import { useState } from 'react'
import { Segmented } from '#/components/Segmented'
import { coats, colors, fonts } from '../tokens.stylex'
import { Avatar, Bar, Check, Collapse, Field, Row, Spinner, Stage, Stream, Text, Window, type Act, type LookTokens } from './kit'
import './assistant.css'

/* What searching with Hinny looks like inside a product, in three scenes on the stageshow kit.
   Mail: someone searches their mailbox; the hybrid results land, each marked with what found it
   (the words, the meaning, or both). Assistant: a notes app's own assistant. The app gives its
   model one tool, search_notes, and that tool's body is a Hinny query (notes.query); the model
   answers from what comes back. The tool is the app's, not part of Hinny: Hinny is the query
   inside it. Memory: a coding agent in a terminal stores what it is told about a project and, in
   a later session, finds it again and acts on it.
   The two chat scenes are the kit's ai-chat example with tool calls added; the people, mails,
   notes and dates are made up for the illustration. */

/** The stage's colours and type are the site's own tokens, so it follows light and dark. */
const TOKENS = {
  '--ss-surface': colors.bg,
  '--ss-canvas': colors.surface,
  '--ss-chrome': colors.surface,
  '--ss-well': colors.surface,
  '--ss-raised': colors.bg,
  '--ss-line': colors.line,
  '--ss-line-faint': colors.line,
  '--ss-text': colors.text,
  '--ss-text-strong': colors.fg,
  '--ss-text-faint': colors.muted,
  '--ss-ink-1': colors.raised,
  '--ss-ink-2': colors.line,
  '--ss-ink-3': colors.lineStrong,
  '--ss-ink-4': colors.faint,
  '--ss-ink-5': colors.muted,
  '--ss-accent': colors.accent,
  '--ss-accent-soft': colors.accentSoft,
  '--ss-font': fonts.sans,
  // The window is outlined by a hairline, not lifted by a shadow: there is no plate behind it.
  '--ss-shadow-window': `0 0 0 1px ${colors.line}`,
  '--hs-mono': fonts.mono,
  '--hs-sun': coats.sun,
} as LookTokens

function Tags({ words, meaning }: { words?: boolean; meaning?: boolean }) {
  return (
    <span className="hs-tags">
      {words && <i className="hs-tag">words</i>}
      {meaning && <i className="hs-tag hs-tag--meaning">meaning</i>}
    </span>
  )
}

/* ---- Mail ---- */

const QUERY = 'tax return filed'
const HITS = [
  { from: 'Dana Okafor', subject: 'Your 2025 return is filed', words: true, meaning: true },
  { from: 'Tax office', subject: 'Confirmation of receipt', meaning: true },
  { from: 'Dana Okafor', subject: 'Re: documents for the return', words: true },
  { from: 'Mira Lindqvist', subject: 'Deadline is the 31st, sent it?', meaning: true },
]

type M = { q: string; focus: boolean; shown: number; picked: boolean }
const mailInitial: M = { q: '', focus: false, shown: 0, picked: false }
const mailPoster: M = { q: QUERY, focus: true, shown: HITS.length, picked: true }

async function mailScript(a: Act<M>) {
  await a.wait(800)
  a.mark('search')
  await a.click('search')
  a.put({ focus: true })
  await a.wait(250)
  await a.type('q', QUERY, { interval: 55 })
  await a.wait(250)
  a.mark('results')
  await a.stagger([1, 2, 3, 4], (n) => a.put({ shown: n }), 130)
  await a.wait(1200)
  a.mark('open')
  await a.click('hit')
  a.put({ picked: true })
  await a.wait(2600)
  a.mark('reset')
  await a.click('clear')
  a.put(mailInitial)
  await a.wait(600)
}

function Mail({ s }: { s: M }) {
  const searching = s.shown > 0
  return (
    <Window
      className="ac-win"
      title="Inbox"
      sidebarWidth={150}
      sidebar={
        <div className="hm-side">
          <div className="hm-box" data-on>
            Inbox
          </div>
          <div className="hm-box">Sent</div>
          <div className="hm-box">Archive</div>
          <Bar w="62%" h={8} r={4} tone={2} style={{ margin: '10px 8px 0' }} />
          <Bar w="48%" h={8} r={4} tone={2} style={{ margin: '8px 8px 0' }} />
        </div>
      }
    >
      <div className="hm-main">
        <div className="hm-search">
          <Field
            value={s.q}
            placeholder="Search mail"
            focused={s.focus}
            box={{ 'data-target': 'search', 'data-cursor': 'ibeam' }}
            trailing={s.q ? <span data-target="clear" data-cursor="pointer" style={{ fontSize: 15, lineHeight: 1, color: 'var(--ss-text-faint)' }}>×</span> : undefined}
          />
        </div>
        <div className="hm-list">
          <div className="hm-set" data-hide={searching || undefined}>
            {[0.7, 0.5, 0.62, 0.44, 0.58].map((w, i) => (
              <Row key={i} lead={<Bar w={28} h={28} r={14} tone={2} />} title={<Bar w={90 + i * 9} h={9} r={4} tone={3} />} subtitle={<Bar w={`${w * 100}%`} h={8} r={4} tone={2} />} />
            ))}
          </div>
          <div className="hm-set" data-hide={!searching || undefined}>
            {HITS.map((h, i) => (
              <div key={h.subject} className="hm-hit" data-show={i < s.shown || undefined}>
                <Row
                  lead={<Avatar name={h.from} size={28} />}
                  title={h.from}
                  subtitle={h.subject}
                  meta={<Tags words={h.words} meaning={h.meaning} />}
                  selected={i === 0 && s.picked}
                  data-target={i === 0 ? 'hit' : undefined}
                  data-cursor="pointer"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </Window>
  )
}

/** A tool call in a thread: the app's own tool, the Hinny call inside it, and what came back.
 * It opens once and stays, as the record of what was done. */
function ToolLine({ state, name, call, done }: { state: 'idle' | 'run' | 'done'; name: string; call: string; done: string }) {
  return (
    <div className="ac-tool" data-done={state === 'done' || undefined}>
      <span className="ac-tool__state">{state === 'done' ? <Check on size={14} /> : <Spinner size={13} />}</span>
      <span className="ac-tool__name">{name}</span>
      <code>{call}</code>
      <span className="ac-tool__done">{done}</span>
    </div>
  )
}

/* ---- Assistant ---- */

const Q = 'What did we decide about the launch date?'
const A = 'You settled on 12 May. The decision is in the week 9 planning note, and Mira’s follow-up moved the press briefing to the 10th.'
const WORDS = A.split(' ').length

type C = { draft: string; sent: boolean; tool: 'idle' | 'run' | 'done'; answer: boolean; words: number; sources: boolean }
const chatInitial: C = { draft: '', sent: false, tool: 'idle', answer: false, words: 0, sources: false }
const chatPoster: C = { draft: '', sent: true, tool: 'done', answer: true, words: WORDS, sources: true }

async function chatScript(a: Act<C>) {
  await a.wait(700)
  a.mark('type')
  await a.type('draft', Q, { interval: 38 })
  await a.wait(400)
  a.mark('send')
  a.put({ sent: true, draft: '' })
  await a.wait(500)
  a.mark('tool')
  a.put({ tool: 'run' })
  await a.wait(1300)
  a.put({ tool: 'done' })
  await a.wait(450)
  a.mark('stream')
  a.put({ answer: true })
  for (let i = 1; i <= WORDS; i++) {
    a.put({ words: i })
    await a.wait(40)
  }
  await a.wait(300)
  a.mark('sources')
  a.put({ sources: true })
  await a.wait(2600)
}

function Assistant({ s }: { s: C }) {
  return (
    <Window title="Notes · Assistant" className="ac-win">
      <div className="ac-body">
        <div className="ac-thread">
          <div className="ac-turn ac-turn--past">
            <Bar w={180} h={30} r={15} tone={1} className="ac-past-q" />
            <Bar w="86%" h={8} r={4} tone={2} />
            <Bar w="62%" h={8} r={4} tone={2} />
          </div>
          <Collapse open={s.sent} reveal="drop" from="below">
            <div className="ac-q">
              <Text>{Q}</Text>
            </div>
          </Collapse>
          {/* The tool call opens once and stays: it is the record of what was searched. */}
          <Collapse open={s.tool !== 'idle'}>
            <ToolLine state={s.tool} name="search_notes" call="notes.query(text=q, vector=embed(q))" done="3 notes" />
          </Collapse>
          <Collapse open={s.answer}>
            <div className="ac-a" data-on>
              <Stream className="ac-stream" text={A} count={s.words} />
              <div className="ac-sources" data-on={s.sources || undefined}>
                <span className="ac-source">Launch planning, week 9</span>
                <span className="ac-source">Re: press briefing</span>
              </div>
            </div>
          </Collapse>
        </div>
        <Field className="ac-composer" value={s.draft} placeholder="Ask about your notes" focused={!s.sent} />
      </div>
    </Window>
  )
}

/* ---- Memory: a coding agent in a terminal ---- */

const NOTE = 'remember: we use pnpm, and tests live beside the code'
const TASK = 'add a test for refunds'
const MEMORIES = ['we use pnpm', 'tests live beside the code']
const CODE = ["test('refunds the original card', async () => {", "  expect((await refund(order)).status).toBe('refunded')", '})']

type R = { saved: boolean; later: boolean; draft: string; asked: boolean; recall: boolean; found: number; wrote: boolean; lines: number; ran: boolean; passed: boolean }
const memInitial: R = { saved: false, later: false, draft: '', asked: false, recall: false, found: 0, wrote: false, lines: 0, ran: false, passed: false }
const memPoster: R = { saved: true, later: true, draft: '', asked: true, recall: true, found: MEMORIES.length, wrote: true, lines: CODE.length, ran: true, passed: true }

// The agent is told two things about the project. Two days later, in a new session, it is asked
// for a test: it recalls both, puts the test beside the code and runs it with pnpm, unprompted.
async function memScript(a: Act<R>) {
  await a.wait(900)
  a.mark('remember')
  a.put({ saved: true })
  await a.wait(1300)
  a.mark('later')
  a.put({ later: true })
  await a.wait(700)
  a.mark('ask')
  await a.type('draft', TASK, { interval: 50 })
  await a.wait(350)
  a.put({ asked: true, draft: '' })
  await a.wait(500)
  a.mark('recall')
  a.put({ recall: true })
  await a.wait(450)
  await a.stagger([1, 2], (n) => a.put({ found: n }), 220)
  await a.wait(700)
  a.mark('write')
  a.put({ wrote: true })
  await a.wait(250)
  await a.stagger([1, 2, 3], (n) => a.put({ lines: n }), 130)
  await a.wait(600)
  a.mark('run')
  a.put({ ran: true })
  await a.wait(900)
  a.put({ passed: true })
  await a.wait(2800)
}

function Step({ name, children }: { name: string; children?: React.ReactNode }) {
  return (
    <div className="tm-l">
      <i className="tm-d" />
      <b>{name}</b>
      {children}
    </div>
  )
}

function Memory({ s }: { s: R }) {
  return (
    <Window chrome="terminal" title="~/billing" className="ac-win hm-term">
      <div className="tm-log">
        <div className="tm-in">
          <span className="tm-p">❯</span>
          <span className="tm-cmd">{NOTE}</span>
        </div>
        <Collapse open={s.saved}>
          <Step name="remember">
            <span className="tm-path">memory.upsert(key, text=note, vector=v)</span>
            <span className="tm-dim">saved</span>
          </Step>
        </Collapse>
        <Collapse open={s.later}>
          <div className="ac-divider tm-div">new session, two days later</div>
        </Collapse>
        <Collapse open={s.asked}>
          <div className="tm-in">
            <span className="tm-p">❯</span>
            <span className="tm-cmd">{TASK}</span>
          </div>
        </Collapse>
        <Collapse open={s.recall}>
          <Step name="recall">
            <span className="tm-path">memory.query(text=q, vector=v)</span>
            <span className="tm-dim">{s.found} found</span>
          </Step>
          {MEMORIES.map((m, i) => (
            <Collapse key={m} open={i < s.found}>
              <div className="tm-mem">· {m}</div>
            </Collapse>
          ))}
        </Collapse>
        <Collapse open={s.wrote}>
          <Step name="Write">
            <span className="tm-path">src/billing/refund.test.ts</span>
          </Step>
          <div className="tm-code">
            {CODE.map((line, i) => (
              <div key={i} className="tm-row" data-show={i < s.lines || undefined}>
                <i>{i + 1}</i>
                <span>{line}</span>
              </div>
            ))}
          </div>
        </Collapse>
        <Collapse open={s.ran}>
          <Step name="Run">
            <span className="tm-path">pnpm test refund</span>
            {s.passed && <span className="tm-ok">1 passed</span>}
          </Step>
        </Collapse>
      </div>
      <div className="tm-box">
        <span className="tm-p">❯</span>
        <span>{s.draft}</span>
        <span className="tm-caret" />
        <span className="tm-right">agent · memory: hinny</span>
      </div>
    </Window>
  )
}

const OPTIONS = [
  { id: 'mail', name: 'Mail' },
  { id: 'assistant', name: 'Assistant' },
  { id: 'memory', name: 'Memory' },
] as const
type Kind = (typeof OPTIONS)[number]['id']

const SIZE = { width: 640, height: 420, compact: { below: 600, width: 358, height: 520 } }

export function SearchShowcase() {
  const [kind, setKind] = useState<Kind>('mail')
  return (
    <div style={{ display: 'grid', gap: 12, minWidth: 0 }}>
      <Segmented label="Example" options={OPTIONS} value={kind} onChange={setKind} />
      {kind === 'mail' ? (
        <Stage key="mail" {...SIZE} initial={mailInitial} poster={mailPoster} script={mailScript} cursor={{ at: { wide: { x: 560, y: 380 }, compact: { x: 300, y: 480 } } }} tokens={TOKENS} className="ac">
          {(s) => <Mail s={s} />}
        </Stage>
      ) : kind === 'assistant' ? (
        <Stage key="assistant" {...SIZE} initial={chatInitial} poster={chatPoster} script={chatScript} play="once" tokens={TOKENS} className="ac">
          {(s) => <Assistant s={s} />}
        </Stage>
      ) : (
        <Stage key="memory" {...SIZE} initial={memInitial} poster={memPoster} script={memScript} play="once" tokens={TOKENS} className="ac">
          {(s) => <Memory s={s} />}
        </Stage>
      )}
    </div>
  )
}
