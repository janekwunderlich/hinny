import * as stylex from '@stylexjs/stylex'
import { useRef, useState } from 'react'
import { GITHUB_URL, INSTALL, RELEASED, UNRELEASED } from '#/data/site'
import { colors, fonts, radii, sizes, space, type } from '../tokens.stylex'
import { Chevron, button } from './Button'
import { CodeTabs, Copy, InlineCode } from './CodeBlock'
import { ExternalLink, Link } from './Link'
import { Callout, P, Strong } from './Prose'
import { Segmented } from './Segmented'
import { focus, link, text } from './styles'

/* Everything the site says about installing Hinny. It all follows RELEASED in data/site.ts:
   until packages are published there is no install command anywhere, only the statement that the
   alpha builds from source, and the docs' steps for doing that. */

const styles = stylex.create({
  actions: { display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: space.x3 },
  status: { display: 'inline-flex', alignItems: 'center', gap: space.x2 },
  ring: { flexShrink: 0, width: '7px', height: '7px', borderWidth: '1.5px', borderStyle: 'solid', borderColor: 'currentColor', borderRadius: '50%' },
  install: { display: 'grid', justifyItems: 'center', gap: space.x3, maxWidth: '100%' },
  command: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.x3,
    minHeight: '44px',
    maxWidth: '100%',
    paddingInlineStart: space.x5,
    paddingInlineEnd: space.x2,
    borderRadius: radii.pill,
    backgroundColor: colors.surface,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
  },
  code: { fontFamily: fonts.mono, fontSize: type.small, color: colors.fg, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  dollar: { color: colors.muted, userSelect: 'none' },
})

/** The home page's actions. Before packages are published: the docs and the benchmarks, and a
 * line saying the alpha builds from source. After: the install command above them. */
export function PrimaryAction() {
  return (
    <>
      {RELEASED && <InstallCommand />}
      <div {...stylex.props(styles.actions)}>
        <Link to="/docs" {...button('primary')}>
          Read the docs
          <Chevron />
        </Link>
        {GITHUB_URL ? (
          <ExternalLink href={GITHUB_URL} {...button('secondary')}>
            View on GitHub
            <Chevron />
          </ExternalLink>
        ) : (
          <Link to="/benchmarks" {...button('secondary')}>
            See the benchmarks
          </Link>
        )}
      </div>
      {!RELEASED && (
        <p {...stylex.props(text.meta, styles.status)}>
          <span aria-hidden="true" {...stylex.props(styles.ring)} />
          {GITHUB_URL ? (
            <ExternalLink href={GITHUB_URL} {...stylex.props(link.quiet, focus.ring)}>
              {UNRELEASED} · v0.3.0
            </ExternalLink>
          ) : (
            UNRELEASED
          )}
        </p>
      )}
    </>
  )
}

/** The install command for one language at a time, with a copy button. Only rendered once
 * RELEASED is true. */
function InstallCommand() {
  const [current, setCurrent] = useState<string>(INSTALL[0].name)
  const ref = useRef<HTMLElement>(null)
  const chosen = INSTALL.find((t) => t.name === current) ?? INSTALL[0]
  return (
    <div {...stylex.props(styles.install)}>
      <Segmented label="Language" options={INSTALL.map((t) => ({ id: t.name, name: t.name }))} value={current} onChange={setCurrent} />
      <div {...stylex.props(styles.command)}>
        <code {...stylex.props(styles.code)}>
          <span {...stylex.props(styles.dollar)}>$ </span>
          <span ref={ref}>{chosen.command}</span>
        </code>
        <Copy from={() => ref.current} />
      </div>
    </div>
  )
}

/** How to get the alpha from the repository, until packages are published. */
const FROM_SOURCE = [
  { name: 'Python', lang: 'shell' as const, code: `git clone ${GITHUB_URL}\npip install ./hinny/bindings/python   # builds the wheel with Rust 1.89+` },
  { name: 'Rust', lang: 'shell' as const, code: `# Cargo.toml\n[dependencies]\nhinny = { git = "${GITHUB_URL}", tag = "v0.3.0" }` },
  { name: 'Node.js', lang: 'shell' as const, code: `git clone ${GITHUB_URL}\ncd hinny/bindings/node && npm ci && npm run build` },
]

/** The docs' install section: the commands once released, the state of things until then. */
export function InstallNotes() {
  if (!RELEASED) {
    return (
      <>
        <Callout>
          <P>
            <Strong>{UNRELEASED}.</Strong> Hinny is an alpha (v0.3.0): APIs and file formats can still change. The <InlineCode>hinny</InlineCode> packages on crates.io, PyPI, and npm are empty <InlineCode>0.0.1</InlineCode> placeholders. Build from{' '}
            {GITHUB_URL ? (
              <ExternalLink href={GITHUB_URL} {...stylex.props(link.inline, focus.ring)}>
                the GitHub repository
              </ExternalLink>
            ) : (
              'source'
            )}
            .
          </P>
        </Callout>
        <P>
          Requires Rust 1.89+. Python wheels build on macOS and Linux for Python 3.11+. The repo also includes the Node.js addon (<InlineCode>bindings/node</InlineCode>), the C library (<InlineCode>crates/hinny-c</InlineCode>, <InlineCode>hinny.h</InlineCode>), and the optional HTTP server (<InlineCode>crates/hinny-server</InlineCode>). MIT or Apache-2.0.
        </P>
        <CodeTabs tabs={FROM_SOURCE} />
      </>
    )
  }
  return (
    <>
      <P>Wheels build for macOS and Linux on Python 3.11+. Dual-licensed under MIT or Apache-2.0.</P>
      <CodeTabs tabs={INSTALL.map((t) => ({ name: t.name, code: t.command }))} />
    </>
  )
}
