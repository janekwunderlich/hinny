import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, media, sizes, space, type } from '../tokens.stylex'

/* Tables: hairlines between rows, no fills, figures aligned. Hinny's own rows carry a small
   square in the accent colour and heavier text; the best value of a column is heavier too.
   `table` holds the styles (the sortable table in BeirTable.tsx uses them with React Aria's
   table); DataTable draws a plain table from arrays. */

export const table = stylex.create({
  /** A table wider than its column scrolls sideways inside this, never the page. */
  scroll: { minWidth: 0, overflowX: 'auto', WebkitOverflowScrolling: 'touch' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: type.small, lineHeight: '20px', fontVariantNumeric: 'tabular-nums' },
  /** For tables with many columns: they keep a readable width and scroll on a phone. */
  wide: { minWidth: '640px' },
  th: {
    paddingBlock: space.x3,
    paddingInlineStart: 0,
    paddingInlineEnd: space.x4,
    textAlign: 'right',
    whiteSpace: 'nowrap',
    fontWeight: 500,
    color: colors.muted,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.lineStrong,
    verticalAlign: 'bottom',
  },
  td: {
    paddingBlock: space.x3,
    paddingInlineStart: 0,
    paddingInlineEnd: space.x4,
    textAlign: 'right',
    whiteSpace: 'nowrap',
    fontWeight: 400,
    color: colors.text,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.line,
    verticalAlign: 'top',
  },
  /** The first column: the row's name, left-aligned, free to wrap. */
  first: { textAlign: 'left', whiteSpace: 'normal', minWidth: '9rem', color: colors.fg },
  last: { paddingInlineEnd: 0 },
  /** The first column stays in place while a wide table scrolls. */
  sticky: { position: 'sticky', left: 0, zIndex: 1, backgroundColor: colors.bg },
  /** Text cells, in tables that compare in words. */
  words: { textAlign: 'left', whiteSpace: 'normal' },
  strong: { fontWeight: 600, color: colors.fg },
  ours: { fontWeight: 500, color: colors.fg },
  /** The tint of Hinny's column, where a table compares by column. */
  tint: { backgroundColor: colors.accentSoft },
  padded: { paddingInlineStart: space.x4 },
  marker: { display: 'inline-block', width: '7px', height: '7px', marginRight: space.x2, backgroundColor: colors.accent, verticalAlign: '1px' },
  sub: { display: 'block', fontSize: type.micro, lineHeight: '16px', fontWeight: 400, color: colors.muted },
  phoneTight: { paddingInlineEnd: { default: space.x4, [media.phone]: space.x3 } },
})

export type Cell = ReactNode | { value: ReactNode; strong?: boolean; sub?: ReactNode }

const isRich = (c: Cell): c is { value: ReactNode; strong?: boolean; sub?: ReactNode } => typeof c === 'object' && c !== null && 'value' in c

/** A plain table. `head` and each row are cells left to right; the first column names the row.
 * `ours` lists the rows that are Hinny's; `tintColumn` tints one column instead (for tables that
 * compare systems by column); `words` sets every cell as wrapping text. */
export function DataTable({
  head,
  rows,
  ours = [],
  tintColumn,
  wide = false,
  words = false,
  label,
}: {
  head: ReactNode[]
  rows: Cell[][]
  ours?: number[]
  tintColumn?: number
  wide?: boolean
  words?: boolean
  label?: string
}) {
  const last = head.length - 1
  return (
    <div {...stylex.props(table.scroll)}>
      <table aria-label={label} {...stylex.props(table.table, wide && table.wide)}>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i} scope="col" {...stylex.props(table.th, i === 0 && table.first, i === 0 && wide && table.sticky, words && table.words, i === last && table.last, i === tintColumn && table.tint, i === tintColumn && table.padded)}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => {
            const mine = ours.includes(r)
            return (
              <tr key={r}>
                {row.map((cell, i) => {
                  const rich = isRich(cell)
                  const value = rich ? cell.value : cell
                  const styles = stylex.props(
                    table.td,
                    i === 0 && table.first,
                    i === 0 && wide && table.sticky,
                    words && table.words,
                    i === last && table.last,
                    mine && table.ours,
                    rich && cell.strong && table.strong,
                    i === tintColumn && table.tint,
                    i === tintColumn && table.padded,
                  )
                  const inner = (
                    <>
                      {i === 0 && mine && <span {...stylex.props(table.marker)} aria-hidden="true" />}
                      {value}
                      {rich && cell.sub && <span {...stylex.props(table.sub)}>{cell.sub}</span>}
                    </>
                  )
                  return i === 0 ? (
                    <th key={i} scope="row" {...styles}>
                      {inner}
                    </th>
                  ) : (
                    <td key={i} {...styles}>
                      {inner}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
