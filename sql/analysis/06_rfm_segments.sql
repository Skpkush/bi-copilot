-- RFM segmentation (Recency, Frequency, Monetary) scored with NTILE quartiles.
-- Snapshot date is the day after the dataset ends (2026-10-01).
WITH base AS (
  SELECT
    o.customer_id,
    julianday('2026-10-01') - julianday(MAX(o.order_date)) AS recency_days,
    COUNT(DISTINCT o.order_id) AS frequency,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS monetary
  FROM orders o
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
  GROUP BY o.customer_id
),
scored AS (
  SELECT
    *,
    NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,  -- most recent = 4
    NTILE(4) OVER (ORDER BY frequency) AS f_score,
    NTILE(4) OVER (ORDER BY monetary) AS m_score
  FROM base
),
segmented AS (
  SELECT
    *,
    CASE
      WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Champions'
      WHEN r_score >= 3 AND f_score >= 2 THEN 'Loyal / Promising'
      WHEN r_score <= 2 AND m_score >= 3 THEN 'At Risk (high value)'
      WHEN r_score = 1 THEN 'Hibernating'
      ELSE 'Needs Attention'
    END AS segment
  FROM scored
)
SELECT
  segment,
  COUNT(*) AS customers,
  ROUND(AVG(recency_days), 0) AS avg_recency_days,
  ROUND(AVG(frequency), 2) AS avg_orders,
  ROUND(SUM(monetary), 2) AS revenue,
  ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1) AS revenue_share_pct
FROM segmented
GROUP BY segment
ORDER BY revenue DESC;
