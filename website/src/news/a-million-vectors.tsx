import * as stylex from '@stylexjs/stylex'
import { BarChart } from '#/components/BarChart'
import { Figure } from '#/components/Figure'
import { Link } from '#/components/Link'
import { H2, Item, List, P } from '#/components/Prose'
import { DataTable } from '#/components/Table'
import { focus, link } from '#/components/styles'
import type { Post } from './index'
import { Cover, InDevelopment, PartsBar, Steps } from './parts'

/* Every number here is from DESIGN.md ("Vectors") or the README's vector table, with the machine
   each was measured on. If those change, change them here too. */

function Body() {
  return (
    <>
      <InDevelopment />
      <P>
        A vector search answers one question: of everything stored, which ten vectors are closest to this one? Comparing the query with every vector answers it exactly, and for a million vectors of 256 dimensions that takes Hinny 13 ms. Its index answers in 0.13 ms and agrees with the exact answer on 99.7% of the neighbours. It gets there by not looking at most of the data, and by looking at the rest in a very small form.
      </P>
      <Figure
        wide
        caption="What one query over a million vectors reads. Synthetic, clustered vectors of 256 dimensions, at the index's calibrated probe budget, recall@10 0.997; counted on an M1 Max under Linux, at the start of the work on the file layout described below."
      >
        <Steps
          steps={[
            { number: '1,000,000', label: 'vectors in the collection' },
            { number: '1,000', label: 'cluster centres ranked' },
            { number: '22,000', label: '1-bit codes scored', share: '2.2% of the collection' },
            { number: '400', label: 'stored vectors read to rerank', share: '0.04%' },
            { number: '10', label: 'results' },
          ]}
        />
      </Figure>

      <H2 id="codes">One bit a dimension</H2>
      <P>
        Hinny keeps every vector twice. Once as a code of one bit a dimension: after a random rotation, the signs of the vector's offset from the centre of its cluster, with two small correction factors. This is RaBitQ. The codes of a million 256-dimension vectors take about 40 MB where the vectors themselves take a gigabyte, so the part of the index a query scans stays in the fast tiers of memory.
      </P>
      <P>
        A code gives an estimate of the distance and, more usefully, a bound on how wrong that estimate can be. The search scores codes, keeps the best results so far, and reads a stored vector only when its bound says it could still belong among them. On the million that is about 400 vectors a query, 0.04% of the collection.
      </P>
      <P>
        The second copy, the one read for that last step, is in half precision. It is read only to rerank, half precision keeps distances within 0.1%, and it halves the file: the index of BEIR FiQA went from 64 to 34 MB at the same recall.
      </P>

      <H2 id="probes">Clusters, and how many to look into</H2>
      <P>
        Vectors are grouped by k-means into about √n clusters, and a cluster's codes are stored together. A query ranks the cluster centres and scans the codes of the nearest few. How many is the setting that decides both speed and recall, and no fixed number is right for all data: probing 16 clusters found 99% of the true neighbours on a mailbox's embeddings, 99.9% on clustered synthetic data and 91% on FiQA's.
      </P>
      <P>
        So each index calibrates itself when it is built. It takes 256 of its own vectors, finds each one's neighbours both exactly and approximately, and stores the smallest budget, from 8 clusters up, that finds 99.7% of their ten nearest. It also stops where more probes stop helping: on the synthetic million, 8 probes already reach 0.996 and it is the codes' estimates, not the probing, that miss the rest, so it keeps 8. Calibration costs about 30 ms for 57,000 vectors.
      </P>
      <Figure
        wide
        caption="Recall@10 against exact search, at each index's calibrated default. Single-threaded queries, warm, on an Apple M5 Pro. A fixed budget of 16 clusters, the default before calibration, gave FiQA 0.913."
      >
        <DataTable
          wide
          label="Vector search at the calibrated default, on three collections"
          head={['Collection', 'Recall@10', 'Median', '99th percentile', 'Exact scan', 'Vectors read to rerank', 'Index']}
          rows={[
            [{ value: 'BEIR FiQA', sub: '57,638 documents, its 648 real questions' }, '0.992', '0.10 ms', '0.22 ms', '0.78 ms', '0.39%', '34 MB'],
            [{ value: 'A mailbox', sub: '23,037 emails, held-out emails as queries' }, '0.998', '0.04 ms', '0.07 ms', '0.31 ms', '0.34%', '14 MB'],
            [{ value: '1 million vectors', sub: 'synthetic, clustered' }, '0.997', '0.13 ms', '0.25 ms', '13 ms', '0.04%', { value: '596 MB', sub: '512 MB of it the stored vectors' }],
          ]}
        />
      </Figure>

      <H2 id="scan">Thirty-two codes at a time</H2>
      <P>
        Scoring a code means adding up the query's values over the code's set bits. Hinny quantizes the query's offset from a cluster's centre to 4 bits a dimension and stores the codes regrouped in blocks of 32, four dimensions to a group. For each probed cluster the query gets a 16-entry table per group, and one SIMD table lookup scores 16 codes on four dimensions at once. This is FastScan, as the RaBitQ authors use it.
      </P>
      <P>
        The sums are the same as scoring code by code, which a test checks, so recall does not move. On an Apple M5 Pro, FiQA's median went from 0.150 to 0.094 ms, and the synthetic million's from 0.21 to 0.13 ms.
      </P>

      <H2 id="layout">A file laid out the way a search reads it</H2>
      <P>
        With scoring that cheap, the profile of one query over the million changed. Of 0.215 ms, about 65 µs went into scoring 22,000 codes and 30 µs into ranking 1,000 cluster centres. The rest, around half of the query, was reading 400 stored vectors to rerank: each one a cache miss somewhere in a 512 MB file.
      </P>
      <Figure boxed caption="One query over the synthetic million before the changes below, 0.215 ms in all. M1 Max under Linux.">
        <PartsBar
          label="Of 215 microseconds: about 65 scoring codes, about 30 ranking cluster centres, the rest reading stored vectors to rerank"
          parts={[
            { name: 'Scoring 22,000 codes', amount: 'about 65 µs', size: 65, tone: 'mid' },
            { name: 'Ranking 1,000 cluster centres', amount: 'about 30 µs', size: 30, tone: 'low' },
            { name: 'Reading 400 stored vectors to rerank', amount: 'the rest', size: 120, tone: 'focus' },
          ]}
        />
      </Figure>
      <P>Three changes followed.</P>
      <List>
        <Item>Stored vectors are fetched a few candidates ahead of being read, so the memory is on its way before it is needed.</Item>
        <Item>The lower bounds of a block's 32 codes are worked out four at a time, and only the codes that pass are estimated one by one. Once there are ten results, nearly none do.</Item>
        <Item>
          The file now holds what a search reads: the codes already in their blocks of 32, the per-vector factors worked out, and the stored vectors in the order of the codes instead of the order of the documents. A search maps the file and builds nothing, reranking needs no lookup from document to row, and a cluster's vectors sit next to each other. The file is 1.4% larger.
        </Item>
      </List>
      <Figure>
        <BarChart
          title="A query over a million vectors, change by change"
          subtitle="Median, synthetic million at its calibrated budget, recall@10 0.997, M1 Max under Linux"
          max={1.2}
          ticks={[0, 0.3, 0.6, 0.9, 1.2]}
          axis="milliseconds, lower is better"
          format={(t) => t.toFixed(1)}
          bars={[
            { name: 'The release before', value: 1.09, label: '1.09 ms' },
            { name: 'At the start of this work', value: 0.215, label: '0.215 ms' },
            { name: 'Vectors fetched ahead', value: 0.163, label: '0.163 ms' },
            { name: 'The file holds what a search reads', value: 0.132, label: '0.132 ms', ours: true },
          ]}
        />
      </Figure>
      <Figure caption="Median query over the synthetic million at its calibrated budget, recall@10 0.997 throughout. M1 Max under Linux. With the last change the 99th percentile went from 0.35 to 0.26 ms.">
        <DataTable
          label="Median query time before and after each change"
          head={['Change', 'Before', 'After']}
          rows={[
            ['Stored vectors fetched ahead of being read', '0.195 ms', '0.163 ms'],
            ['The file holds what a search reads', '0.163 ms', '0.132 ms'],
            ['Altogether, from the start of this work', '0.215 ms', { value: '0.132 ms', strong: true }],
            ['From the release before, which counted deletions in every search', '1.09 ms', { value: '0.132 ms', strong: true }],
          ]}
        />
      </Figure>

      <H2 id="generations">One set of clusters for the whole index</H2>
      <P>
        Hinny stores data in immutable segments that are merged in the background, and at first each segment had clusters of its own. A search then pays for every segment: S segments scan about √S times the codes of one merged segment. The million as writing leaves it, in 8 segments, scanned 5.7% of the codes against 2.2% merged into one, and took 0.26 against 0.13 ms.
      </P>
      <P>
        Segments now share their cluster centres. Writing a new segment assigns its vectors to the centres the namespace already has, with no clustering and no calibration. A search ranks the centres once and scans every segment's list for a probed cluster into one top ten. A merge copies the lists as they are, so it changes no answer. The centres are trained for twice the vectors there were, and trained again once there are four times as many.
      </P>
      <P>
        The million as writing leaves it is now 5 segments: 2.6% of the codes scanned, 0.15 ms at recall@10 0.996, and 49,000 searches a second on 8 threads, where there had been 10,500. These last figures were measured while other work ran on the machine.
      </P>

      <H2 id="not">What did not help</H2>
      <List>
        <Item>Probing every cluster within a factor of the nearest centre's distance, a rule that adapts to the query, traded recall for work no better than a fixed budget on FiQA, with a longer tail.</Item>
        <Item>Stopping once the top ten had not changed for 48 clusters gave FiQA 0.990 at 0.10 ms against 0.993 at 0.15 ms: a little better than a smaller fixed budget, but below the target the budget is calibrated for.</Item>
        <Item>Putting each vector in a second cluster, simulated on FiQA with exact distances: 0.2 to 1.3 points of recall more at equal work, for twice the codes.</Item>
      </List>

      <H2 id="next">What is next</H2>
      <P>
        Ranking the cluster centres is now a visible share of a query, and a larger collection would want more of them. The plan is to rank centres through 1-bit codes of their own, and then to use more clusters. From the measured costs, 22 ns a centre and 3 ns a code, the index's part of a search should fall from about 95 µs to between 35 and 50 µs. That is an estimate, not a measurement.
      </P>
      <P>
        Also open: clusters are retrained when a namespace grows but not yet when its data drifts, and the x86-64 versions of these kernels pass the test suite under emulation but have not been timed on an x86 processor. The current numbers for full text, vectors and hybrid search are on the{' '}
        <Link to="/benchmarks" {...stylex.props(link.inline, focus.ring)}>
          benchmarks page
        </Link>
        .
      </P>
    </>
  )
}

export const vectors: Post = {
  slug: 'a-million-vectors',
  title: 'A million vectors in 0.13 ms',
  short: 'A million vectors in 0.13 ms',
  date: '2026-11-13',
  category: 'Engineering',
  coat: 'iris',
  summary: 'How Hinny finds the top 10 nearest of a million vectors in 0.13 ms on a single thread: 1-bit RaBitQ codes, self-calibrated probe budgets, and a segment layout built for sequential scans.',
  lede: 'Hinny finds the 10 nearest of a million 256-dimensional vectors in 0.13 ms at the median, single-threaded, at 0.997 recall@10. Here is where that time goes and what it took to get there.',
  Cover: () => (
    <Cover
      coat="iris"
      lines={[
        ['vectors', '1,000,000'],
        ['codes scored', '22,000'],
        ['vectors read to rerank', '400'],
        ['recall@10', '0.997'],
        ['median query', '0.13 ms'],
      ]}
    />
  ),
  Body,
}
