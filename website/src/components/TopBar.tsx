import * as stylex from '@stylexjs/stylex'
import { useRouterState } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { Button, Dialog, DialogTrigger, Disclosure, DisclosurePanel, Heading, Menu, MenuTrigger, Modal, ModalOverlay, Popover } from 'react-aria-components'
import { NAV, type NavItem, type NavLink } from '#/data/nav'
import { ANNOUNCE, GITHUB_URL, NAME, UNRELEASED } from '#/data/site'
import { colors, media, radii, sizes, space, type } from '../tokens.stylex'
import { Logo } from './Brand'
import { Caret, button } from './Button'
import { ExternalLink, Link, MenuItemLink } from './Link'
import { focus, frame, text } from './styles'

/* The top bar: the logo, a few text links (a group of links is a menu), one pill. It stays at
   the top of the window over a blurred veil of the page. Below 860px the links fold into a menu
   that opens as a sheet. Menus, the sheet and its sections are React Aria's, so they close on
   Escape, trap and return focus, and work from the keyboard. */

const fade = stylex.keyframes({ from: { opacity: 0, transform: 'translateY(-4px)' }, to: { opacity: 1, transform: 'none' } })
const veilIn = stylex.keyframes({ from: { opacity: 0 }, to: { opacity: 1 } })

const styles = stylex.create({
  bar: {
    position: 'sticky',
    top: 0,
    zIndex: 50,
    backgroundColor: colors.veil,
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    paddingTop: 'env(safe-area-inset-top, 0px)',
  },
  row: { display: 'flex', alignItems: 'center', gap: space.x6, height: sizes.bar },
  brand: { display: 'inline-flex', alignItems: 'center', textDecorationLine: 'none', marginRight: space.x2 },
  links: { display: { default: 'flex', [media.compact]: 'none' }, alignItems: 'center', gap: space.x1 },
  item: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    paddingBlock: '6px',
    paddingInline: space.x3,
    borderRadius: radii.pill,
    borderWidth: 0,
    backgroundColor: { default: 'transparent', ':hover': colors.raised },
    fontSize: type.meta,
    lineHeight: '20px',
    fontWeight: 500,
    color: colors.fg,
    textDecorationLine: 'none',
    cursor: 'pointer',
    transitionProperty: 'background-color',
    transitionDuration: '150ms',
  },
  itemOpen: { backgroundColor: colors.raised },
  end: { marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: space.x2 },
  wideOnly: { display: { default: 'inline-flex', [media.compact]: 'none' } },
  compactOnly: { display: { default: 'none', [media.compact]: 'inline-flex' } },

  // A menu under a group of the bar.
  popover: {
    backgroundColor: colors.bg,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: radii.card,
    boxShadow: '0 12px 32px rgba(0, 0, 0, 0.14)',
    animationName: fade,
    animationDuration: '140ms',
    animationTimingFunction: 'ease-out',
  },
  menu: { display: 'grid', gap: '2px', padding: '6px', width: '320px', maxWidth: 'calc(100vw - 32px)', outlineStyle: 'none' },
  menuItem: {
    display: 'grid',
    gap: '2px',
    paddingBlock: space.x2,
    paddingInline: space.x3,
    borderRadius: '8px',
    textDecorationLine: 'none',
    cursor: 'pointer',
    outlineStyle: 'none',
    backgroundColor: { default: 'transparent', ':hover': colors.raised, ':focus': colors.raised },
  },
  menuLabel: { fontSize: type.small, lineHeight: '20px', fontWeight: 500, color: colors.fg },
  menuNote: { fontSize: type.meta, lineHeight: '18px', color: colors.muted },

  // The phone menu: a sheet over the page.
  menuButton: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '36px',
    height: '36px',
    borderRadius: radii.pill,
    borderWidth: 0,
    backgroundColor: { default: colors.raised, ':hover': colors.raisedHover },
    color: colors.fg,
    cursor: 'pointer',
  },
  overlay: { position: 'fixed', inset: 0, zIndex: 100, backgroundColor: colors.bg, overflowY: 'auto', animationName: veilIn, animationDuration: '140ms' },
  sheet: { minHeight: '100%', outlineStyle: 'none' },
  dialog: { outlineStyle: 'none', paddingBottom: space.x12 },
  sheetNav: { display: 'grid', paddingTop: space.x4 },
  sheetRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    paddingBlock: space.x4,
    paddingInline: 0,
    borderWidth: 0,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.line,
    backgroundColor: 'transparent',
    fontSize: '20px',
    lineHeight: '28px',
    fontWeight: 500,
    letterSpacing: '-0.015em',
    color: colors.fg,
    textAlign: 'left',
    textDecorationLine: 'none',
    cursor: 'pointer',
  },
  sheetHeading: { margin: 0, fontSize: 'inherit', fontWeight: 'inherit' },
  caret: { transitionProperty: 'transform', transitionDuration: '150ms' },
  caretOpen: { transform: 'rotate(180deg)' },
  sheetPanel: { display: 'grid', borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line, paddingBlock: space.x2 },
  sheetLink: { display: 'block', paddingBlock: space.x3, fontSize: type.body, lineHeight: '24px', color: colors.text, textDecorationLine: 'none' },
  sheetActions: { display: 'grid', gap: space.x3, paddingTop: space.x8 },
})

const isGroup = (item: NavItem): item is { label: string; items: NavLink[] } => 'items' in item

/** The pill at the right end: the docs, in every state of the project. */
function Action({ full = false }: { full?: boolean }) {
  return (
    <Link to="/docs" {...button('primary', full ? 'md' : 'sm', full)}>
      Read the docs
    </Link>
  )
}

export function TopBar() {
  return (
    <header {...stylex.props(styles.bar)}>
      <div {...stylex.props(frame.page, styles.row)}>
        <Link to="/" aria-label={`${NAME}, home`} {...stylex.props(styles.brand, focus.ring)}>
          <Logo />
        </Link>
        {ANNOUNCE && GITHUB_URL && (
          <div {...stylex.props(styles.end)}>
            <ExternalLink href={GITHUB_URL} {...button('secondary', 'sm')}>
              GitHub
            </ExternalLink>
          </div>
        )}
        {!ANNOUNCE && (
          <>
        <nav aria-label="Site" {...stylex.props(styles.links)}>
          {NAV.map((item) =>
            isGroup(item) ? (
              <MenuTrigger key={item.label}>
                <Button className={({ isPressed }) => stylex.props(styles.item, focus.ringOnly, isPressed && styles.itemOpen).className ?? ''}>
                  {item.label}
                  <Caret />
                </Button>
                <Popover placement="bottom start" offset={8} {...stylex.props(styles.popover)}>
                  <Menu aria-label={item.label} {...stylex.props(styles.menu)}>
                    {item.items.map((l) => (
                      <MenuItemLink key={l.label} to={l.to} hash={l.hash} textValue={l.label} {...stylex.props(styles.menuItem)}>
                        <span {...stylex.props(styles.menuLabel)}>{l.label}</span>
                        {l.note && <span {...stylex.props(styles.menuNote)}>{l.note}</span>}
                      </MenuItemLink>
                    ))}
                  </Menu>
                </Popover>
              </MenuTrigger>
            ) : (
              <Link key={item.label} to={item.to} hash={item.hash} {...stylex.props(styles.item, focus.ringOnly)}>
                {item.label}
              </Link>
            ),
          )}
        </nav>
        <div {...stylex.props(styles.end)}>
          {GITHUB_URL && (
            <span {...stylex.props(styles.wideOnly)}>
              <ExternalLink href={GITHUB_URL} {...button('secondary', 'sm')}>
                GitHub
              </ExternalLink>
            </span>
          )}
          <Action />
          <PhoneMenu />
        </div>
          </>
        )}
      </div>
    </header>
  )
}

/** The links as a sheet over the page, for narrow windows. Groups are sections that fold. */
function PhoneMenu() {
  const [open, setOpen] = useState(false)
  // Following a link closes it, including a link to another part of the page it is on.
  const where = useRouterState({ select: (s) => s.location.href })
  useEffect(() => setOpen(false), [where])
  const close = () => setOpen(false)
  return (
    <DialogTrigger isOpen={open} onOpenChange={setOpen}>
      <Button aria-label="Menu" {...stylex.props(styles.menuButton, styles.compactOnly, focus.ringOnly)}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
          <path d="M2.5 5.5h11M2.5 10.5h11" />
        </svg>
      </Button>
      <ModalOverlay isDismissable {...stylex.props(styles.overlay)}>
        <Modal {...stylex.props(styles.sheet)}>
          <Dialog aria-label="Site menu" {...stylex.props(styles.dialog)}>
            <div {...stylex.props(frame.page)}>
              <div {...stylex.props(styles.row)}>
                <Link to="/" aria-label={`${NAME}, home`} onClick={close} {...stylex.props(styles.brand, focus.ring)}>
                  <Logo />
                </Link>
                <div {...stylex.props(styles.end)}>
                  <Button aria-label="Close the menu" onPress={close} {...stylex.props(styles.menuButton, styles.compactOnly, focus.ringOnly)}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
                      <path d="m3.5 3.5 9 9M12.5 3.5l-9 9" />
                    </svg>
                  </Button>
                </div>
              </div>
              <nav aria-label="Site" {...stylex.props(styles.sheetNav)}>
                {NAV.map((item) =>
                  isGroup(item) ? (
                    <Disclosure key={item.label} defaultExpanded>
                      {({ isExpanded }) => (
                        <>
                          <Heading level={2} {...stylex.props(styles.sheetHeading)}>
                            <Button slot="trigger" {...stylex.props(styles.sheetRow, focus.ring)}>
                              {item.label}
                              <span {...stylex.props(styles.caret, isExpanded && styles.caretOpen)}>
                                <Caret size={14} />
                              </span>
                            </Button>
                          </Heading>
                          <DisclosurePanel>
                            <div {...stylex.props(styles.sheetPanel)}>
                              {item.items.map((l) => (
                                <Link key={l.label} to={l.to} hash={l.hash} onClick={close} {...stylex.props(styles.sheetLink, focus.ring)}>
                                  {l.label}
                                </Link>
                              ))}
                            </div>
                          </DisclosurePanel>
                        </>
                      )}
                    </Disclosure>
                  ) : (
                    <Link key={item.label} to={item.to} hash={item.hash} onClick={close} {...stylex.props(styles.sheetRow, focus.ring)}>
                      {item.label}
                    </Link>
                  ),
                )}
              </nav>
              <div {...stylex.props(styles.sheetActions)}>
                <span onClick={close}>
                  <Action full />
                </span>
                {GITHUB_URL && (
                  <ExternalLink href={GITHUB_URL} {...button('secondary', 'md', true)}>
                    GitHub
                  </ExternalLink>
                )}
                <p {...stylex.props(text.meta, text.center)}>{UNRELEASED}</p>
              </div>
            </div>
          </Dialog>
        </Modal>
      </ModalOverlay>
    </DialogTrigger>
  )
}
