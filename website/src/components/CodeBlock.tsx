import * as stylex from '@stylexjs/stylex'
import { useRef, useState, type ReactNode } from 'react'
import { Button, Tab, TabList, TabPanel, Tabs } from 'react-aria-components'
import { colors, fonts, radii, sizes, space, type } from '../tokens.stylex'
import { highlight, type Lang } from './highlight'
import { pill } from './Segmented'
import { focus } from './styles'

/* Code: a panel a step off the page, with a bar above it that holds the language tabs (or a
   title) and a copy button. The tabs are React Aria's: arrow keys move between them. */

const styles = stylex.create({
  panel: {
    minWidth: 0,
    backgroundColor: colors.surface,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: radii.card,
    textAlign: 'left',
    overflow: 'hidden',
  },
  bar: {
    display: 'flex',
    alignItems: 'center',
    gap: space.x3,
    minHeight: '48px',
    paddingInline: space.x3,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.line,
  },
  tabs: { display: 'flex', gap: space.x1 },
  title: { paddingInline: space.x2, fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  note: { marginLeft: 'auto', fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  copy: {
    marginLeft: 'auto',
    minHeight: '32px',
    paddingInline: space.x3,
    borderRadius: radii.pill,
    borderWidth: 0,
    backgroundColor: { default: 'transparent', ':hover': colors.raised },
    fontSize: type.meta,
    lineHeight: '20px',
    fontWeight: 500,
    color: colors.muted,
    cursor: 'pointer',
  },
  copyAfterNote: { marginLeft: 0 },
  pre: {
    margin: 0,
    paddingBlock: space.x5,
    paddingInline: space.x5,
    overflowX: 'auto',
    fontFamily: fonts.mono,
    fontSize: type.code,
    lineHeight: 1.7,
    fontVariantLigatures: 'none',
    tabSize: 4,
    color: colors.text,
    outlineStyle: 'none',
  },
  panelFocus: { outlineStyle: 'none' },
  inline: {
    fontFamily: fonts.mono,
    fontSize: '0.875em',
    fontVariantLigatures: 'none',
    color: colors.fg,
    backgroundColor: colors.raised,
    paddingBlock: '0.1em',
    paddingInline: '0.35em',
    borderRadius: '5px',
    overflowWrap: 'anywhere',
  },
})

const LANGS: Record<string, Lang> = { Python: 'python', Rust: 'rust', 'Node.js': 'js', C: 'c' }
/** Code given as text is coloured for its language; anything else is shown as it is. */
const show = (code: ReactNode, lang: Lang) => (typeof code === 'string' ? highlight(code, lang) : code)

/** Code samples in tabs (Python, Rust, C) under a bar with a copy button. A tab's language is
 * its name's, unless it says otherwise. */
export function CodeTabs({ tabs, note, label = 'Language' }: { tabs: { name: string; code: ReactNode; lang?: Lang }[]; note?: string; label?: string }) {
  const body = useRef<HTMLDivElement>(null)
  if (tabs.length === 1) {
    return (
      <Code title={tabs[0]!.name} note={note} lang={tabs[0]!.lang ?? LANGS[tabs[0]!.name] ?? 'shell'}>
        {tabs[0]!.code}
      </Code>
    )
  }
  return (
    <Tabs {...stylex.props(styles.panel)}>
      <div {...stylex.props(styles.bar)}>
        <TabList aria-label={label} {...stylex.props(styles.tabs)}>
          {tabs.map((t) => (
            <Tab key={t.name} id={t.name} className={({ isSelected }) => stylex.props(pill.base, isSelected && pill.on, focus.ringOnly).className ?? ''}>
              {t.name}
            </Tab>
          ))}
        </TabList>
        {note && <span {...stylex.props(styles.note)}>{note}</span>}
        <Copy from={() => body.current?.querySelector('pre')} afterNote={!!note} />
      </div>
      <div ref={body}>
        {tabs.map((t) => (
          <TabPanel key={t.name} id={t.name} {...stylex.props(styles.panelFocus)}>
            <pre {...stylex.props(styles.pre)} tabIndex={0}>
              {show(t.code, t.lang ?? LANGS[t.name] ?? 'shell')}
            </pre>
          </TabPanel>
        ))}
      </div>
    </Tabs>
  )
}

/** One code block with a title and a copy button. */
export function Code({ children, title = 'shell', note, lang = 'shell' }: { children: ReactNode; title?: string; note?: string; lang?: Lang }) {
  const ref = useRef<HTMLPreElement>(null)
  return (
    <div {...stylex.props(styles.panel)}>
      <div {...stylex.props(styles.bar)}>
        <span {...stylex.props(styles.title)}>{title}</span>
        {note && <span {...stylex.props(styles.note)}>{note}</span>}
        <Copy from={() => ref.current} afterNote={!!note} />
      </div>
      <pre ref={ref} {...stylex.props(styles.pre)} tabIndex={0}>
        {show(children, lang)}
      </pre>
    </div>
  )
}

/** Copies an element's text; where there is no clipboard, selects it instead. */
export function Copy({ from, afterNote = false }: { from: () => Element | null | undefined; afterNote?: boolean }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    const el = from()
    const text = el instanceof HTMLElement ? el.innerText : ''
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // No clipboard: select the code, so the reader can copy it.
      const range = document.createRange()
      if (el) range.selectNodeContents(el)
      getSelection()?.removeAllRanges()
      getSelection()?.addRange(range)
    }
  }
  return (
    <Button onPress={copy} {...stylex.props(styles.copy, afterNote && styles.copyAfterNote, focus.ringOnly)}>
      <span aria-live="polite">{copied ? 'Copied' : 'Copy'}</span>
    </Button>
  )
}

/** Code inside a sentence. */
export const InlineCode = ({ children }: { children: ReactNode }) => <code {...stylex.props(styles.inline)}>{children}</code>
