import * as stylex from '@stylexjs/stylex'
import { ToggleButton, ToggleButtonGroup } from 'react-aria-components'
import { colors, radii, space, type } from '../tokens.stylex'
import { focus } from './styles'

/* A row of small pills of which one is chosen: the control over a chart or a table. React Aria's
   toggle group with exactly one selected, so it is a radio group to the keyboard (arrow keys). */

export const pill = stylex.create({
  group: { display: 'inline-flex', flexWrap: 'wrap', gap: space.x1 },
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    minHeight: '32px',
    paddingInline: space.x3,
    borderRadius: radii.pill,
    borderWidth: 0,
    backgroundColor: 'transparent',
    fontSize: type.meta,
    lineHeight: '20px',
    fontWeight: 500,
    color: { default: colors.muted, ':hover': colors.fg },
    cursor: 'pointer',
    transitionProperty: 'background-color, color',
    transitionDuration: '150ms',
  },
  on: { backgroundColor: colors.raised, color: colors.fg },
})

export function Segmented<T extends string>({ label, options, value, onChange }: { label: string; options: readonly { id: T; name: string }[]; value: T; onChange: (value: T) => void }) {
  return (
    <ToggleButtonGroup
      aria-label={label}
      selectionMode="single"
      disallowEmptySelection
      selectedKeys={[value]}
      onSelectionChange={(keys) => {
        const [key] = [...keys]
        if (key !== undefined) onChange(key as T)
      }}
      {...stylex.props(pill.group)}
    >
      {options.map((o) => (
        <ToggleButton key={o.id} id={o.id} className={({ isSelected }) => stylex.props(pill.base, isSelected && pill.on, focus.ringOnly).className ?? ''}>
          {o.name}
        </ToggleButton>
      ))}
    </ToggleButtonGroup>
  )
}
