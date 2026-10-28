-- Revenue by region and loyalty tier (latest quarter), with each tier's share
-- of its region. Techniques: three-way join, partitioned window share.
WITH mix AS (
  SELECT
    o.region,
    c.loyalty_tier,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue
  FROM orders o
  JOIN customers c ON c.customer_id = o.customer_id
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
    AND o.region IS NOT NULL
    AND o.order_date BETWEEN '2026-07-01' AND '2026-09-30'
  GROUP BY o.region, c.loyalty_tier
)
SELECT
  region,
  loyalty_tier,
  ROUND(revenue, 2) AS revenue,
  ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY region), 1) AS share_of_region_pct
FROM mix
ORDER BY region, revenue DESC;
