-- Does discounting buy volume, or just give margin away?
-- Compares order size, order value and margin across discount bands.
SELECT
  CASE
    WHEN o.discount = 0 THEN '0%'
    WHEN o.discount <= 0.10 THEN '1-10%'
    WHEN o.discount <= 0.15 THEN '11-15%'
    ELSE 'Over 15%'
  END AS discount_band,
  COUNT(*) AS orders,
  ROUND(AVG(o.quantity), 2) AS avg_units_per_order,
  ROUND(AVG(o.quantity * o.unit_price * (1 - o.discount)), 2) AS avg_order_value,
  ROUND(100.0 * SUM(o.quantity * o.unit_price * (1 - o.discount) - o.quantity * p.cost_price)
        / NULLIF(SUM(o.quantity * o.unit_price * (1 - o.discount)), 0), 1) AS gross_margin_pct
FROM orders o
JOIN products p ON p.product_id = o.product_id
WHERE o.status NOT IN ('Cancelled', 'Returned')
  AND o.unit_price IS NOT NULL
GROUP BY discount_band
ORDER BY MIN(o.discount);
