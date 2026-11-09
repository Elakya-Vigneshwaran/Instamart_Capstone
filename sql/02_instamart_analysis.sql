
-- INSTAMART DATABASE ANALYSIS

-- 1. Display all database tables
SELECT name
FROM sqlite_master
WHERE type = 'table';

-- 2. View first five customers
SELECT *
FROM customers
LIMIT 5;

-- 3. View first five orders
SELECT *
FROM orders
LIMIT 5;

-- 4. Count total customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 5. Count total orders
SELECT COUNT(*) AS total_orders
FROM orders;

-- 6. Calculate total revenue from completed orders
SELECT
    SUM(order_amount) AS total_revenue
FROM orders
WHERE status = 'completed';

-- 7. Calculate revenue by city
SELECT
    c.city,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY total_revenue DESC;

-- 8. Top 10 customers by spending
SELECT
    c.customer_id,
    c.city,
    ROUND(SUM(o.order_amount), 2) AS total_spending
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY total_spending DESC
LIMIT 10;

-- 9. Calculate average delivery time
SELECT
    ROUND(AVG(delivery_minutes), 2) AS average_delivery_time
FROM orders
WHERE status = 'completed'
AND delivery_minutes IS NOT NULL
AND delivery_minutes >= 0;

-- 10. Count cancelled orders
SELECT
    COUNT(*) AS total_cancelled_orders
FROM orders
WHERE status = 'cancelled';


-- 11. Calculate monthly revenue
SELECT
    strftime('%Y-%m', order_ts) AS order_month,
    ROUND(SUM(order_amount), 2) AS monthly_revenue
FROM orders
WHERE status = 'completed'
AND order_amount IS NOT NULL
GROUP BY order_month
ORDER BY order_month;


-- 12. Count orders by status
SELECT
    status,
    COUNT(*) AS total_orders
FROM orders
GROUP BY status
ORDER BY total_orders DESC;