RESUME_EXAMPLES = [
    {
        "input": """
        Sarah Ahmed
        Email: sarah@example.com
        Software Engineer at ABC Technologies.
        Skills: Python, Django, PostgreSQL.
        BSc in Computer Science from XYZ University, 2023.
        """,
        "guidance": """
        Extract the candidate's name, email, professional experience,
        technical skills, and education. Do not invent missing details.
        """
    }
]


INVOICE_EXAMPLES = [
    {
        "input": """
        Invoice Number: INV-1001
        Date: 15 January 2026
        Seller: Tech Solutions
        Buyer: Global Services

        Item: Software Development
        Quantity: 2
        Unit Price: 1000 USD
        Total: 2000 USD

        Subtotal: 2000 USD
        Tax: 200 USD
        Grand Total: 2200 USD
        """,
        "guidance": """
        Extract invoice metadata, seller and buyer names, line items,
        currency, subtotal, tax, and grand total. Preserve the provided
        values and do not calculate missing amounts.
        """
    }
]