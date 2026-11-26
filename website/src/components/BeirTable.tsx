import * as stylex from '@stylexjs/stylex'
import { useMemo, useState } from 'react'
import { Cell, Column, Row, Table, TableBody, TableHeader, type SortDescriptor } from 'react-aria-components'
import { configurations, datasets, kinds, ms, results, type Kind } from '#/data/beir'
import { colors, space } from '../tokens.stylex'
import { chart } from './ChartParts'
import { Segmented } from './Segmented'
import { table } from './Table'
import { show, text } from './styles'

type Metric = 'ndcg10' | 'recall100' | 'p50_ms'
const metrics: { id: Metric; name: string; better: 'high' | 'low' }[] = [
  { id: 'ndcg10', name: 'nDCG@10', better: 'high' },
  { id: 'recall100', name: 'Recall@100', better: 'high' },
  { id: 'p50_ms', name: 'p50, top 10', better: 'low' },
]

type Line = { name: string; ours: boolean; values: Partial<Record<string, number>> }

const styles = stylex.create({
  wrap: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: space.x4, minWidth: 0 },
  column: { cursor: 'pointer', userSelect: 'none', outlineStyle: 'solid', outlineColor: colors.fg, outlineOffset: '-2px', outlineWidth: { default: '0px', ':focus-visible': '2px' }, color: { default: colors.muted, ':hover': colors.fg } },
  cell: { outlineStyle: 'solid', outlineColor: colors.fg, outlineOffset: '-2px', outlineWidth: { default: '0px', ':focus-visible': '2px' } },
  row: { outlineStyle: 'none' },
  arrow: { display: 'inline-block', width: '1em', textAlign: 'right', color: colors.fg },
  best: { fontWeight: 600, color: colors.fg, textDecorationLine: 'underline', textDecorationThickness: '1.5px', textUnderlineOffset: '3px' },
})

/** Every configuration of one kind across the five datasets, sortable by any column. The table
 * is React Aria's: a column header sorts on click or Enter, and arrow keys move through cells. */
export function BeirTable() {
  const [kind, setKind] = useState<Kind>('hybrid')
  const [metric, setMetric] = useState<Metric>('ndcg10')
  const [sort, setSort] = useState<SortDescriptor | null>(null)
  const m = metrics.find((x) => x.id === metric)!

  const data = useMemo<Line[]>(
    () =>
      configurations
        .filter((c) => c.kind === kind)
        .map((c) => {
          const line: Line = { name: c.name, ours: c.ours, values: {} }
          for (const d of datasets) {
            const r = results.find((r) => r.dataset === d.id && r.system === c.system && r.mode === c.mode)
            if (r) line.values[d.id] = r[metric]
          }
          return line
        }),
    [kind, metric],
  )
  // The best of each column, for bold.
  const best = useMemo(() => {
    const out: Record<string, number> = {}
    for (const d of datasets) {
      const vs = data.map((l) => l.values[d.id]).filter((v): v is number => v != null)
      out[d.id] = m.better === 'high' ? Math.max(...vs) : Math.min(...vs)
    }
    return out
  }, [data, m])

  const rows = useMemo(() => {
    if (!sort) return data
    const sign = sort.direction === 'descending' ? -1 : 1
    const column = String(sort.column)
    return [...data].sort((a, b) => {
      if (column === 'name') return sign * a.name.localeCompare(b.name, 'en', { numeric: true })
      const x = a.values[column]
      const y = b.values[column]
      // Rows without a value go last, whichever way the column is sorted.
      if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1
      return sign * (x - y)
    })
  }, [data, sort])

  // A column not sorted yet starts with its best value first: high for quality, low for time.
  const onSort = (next: SortDescriptor) => {
    const fresh = next.column !== sort?.column
    setSort(fresh && next.column !== 'name' && m.better === 'high' ? { column: next.column, direction: 'descending' } : next)
  }

  const fmt = (v: number) => (metric === 'p50_ms' ? ms(v) : v.toFixed(3))
  const last = datasets.length

  return (
    <div {...stylex.props(styles.wrap)}>
      <div {...stylex.props(chart.controls)}>
        <Segmented label="Kind of search" options={kinds} value={kind} onChange={setKind} />
        <Segmented label="Measure" options={metrics} value={metric} onChange={setMetric} />
      </div>
      <div {...stylex.props(table.scroll)}>
        <Table aria-label={`${m.name} of every ${kind} configuration on five BEIR datasets`} sortDescriptor={sort ?? undefined} onSortChange={onSort} {...stylex.props(table.table, table.wide)}>
          <TableHeader>
            <Column id="name" isRowHeader allowsSorting {...stylex.props(table.th, table.first, table.sticky, styles.column)}>
              {({ sortDirection }) => (
                <>
                  System
                  <span aria-hidden="true" {...stylex.props(styles.arrow)}>
                    {sortDirection === 'ascending' ? '↑' : sortDirection === 'descending' ? '↓' : ''}
                  </span>
                </>
              )}
            </Column>
            {datasets.map((d, i) => (
              <Column key={d.id} id={d.id} allowsSorting {...stylex.props(table.th, styles.column, i === last - 1 && table.last)}>
                {({ sortDirection }) => (
                  <>
                    {d.name}
                    <span aria-hidden="true" {...stylex.props(styles.arrow)}>
                      {sortDirection === 'ascending' ? '↑' : sortDirection === 'descending' ? '↓' : ''}
                    </span>
                  </>
                )}
              </Column>
            ))}
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <Row key={row.name} id={row.name} {...stylex.props(styles.row)}>
                <Cell {...stylex.props(table.td, table.first, table.sticky, styles.cell, row.ours && table.ours)}>
                  {row.ours && <span aria-hidden="true" {...stylex.props(table.marker)} />}
                  {row.name}
                </Cell>
                {datasets.map((d, i) => {
                  const v = row.values[d.id]
                  const value = v == null ? '' : fmt(v)
                  const isBest = v != null && fmt(best[d.id]!) === value
                  return (
                    <Cell key={d.id} {...stylex.props(table.td, styles.cell, row.ours && table.ours, i === last - 1 && table.last)}>
                      {isBest ? <strong {...stylex.props(styles.best)}>{value}</strong> : value}
                    </Cell>
                  )
                })}
              </Row>
            ))}
          </TableBody>
        </Table>
      </div>
      <p {...stylex.props(text.small)}>
        {m.name}, {m.better === 'high' ? 'higher is better' : 'lower is better'}. Best per dataset in bold. Sort by any column<span {...stylex.props(show.phone)}>; swipe sideways for the other datasets</span>.
      </p>
    </div>
  )
}
