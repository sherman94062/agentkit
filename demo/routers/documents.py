"""Documents router — PDF upload and listing."""

import json
from pathlib import Path

import pymupdf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from demo.db import client as db

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    address: str = Form(...),
    doc_type: str = Form("inspection"),
):
    """Upload a PDF document and extract text with PyMuPDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()

    # Save file to disk
    file_path = UPLOAD_DIR / file.filename
    file_path.write_bytes(content)

    # Extract text with PyMuPDF
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append({"page": page_num + 1, "text": text})
        full_text = "\n\n".join(p["text"] for p in pages)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    # Save to database
    doc_id = db.save_document(
        address=address,
        filename=file.filename,
        doc_type=doc_type,
        content_text=full_text,
        extracted_json={"pages": len(pages), "char_count": len(full_text)},
    )

    return {
        "id": doc_id,
        "filename": file.filename,
        "address": address,
        "doc_type": doc_type,
        "pages": len(pages),
        "chars_extracted": len(full_text),
    }


@router.get("")
async def list_documents(address: str = ""):
    """List uploaded documents, optionally filtered by address."""
    docs = db.get_documents(address)
    return {
        "documents": [
            {
                "id": d["id"],
                "address": d["address"],
                "filename": d["filename"],
                "doc_type": d["doc_type"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in docs
        ]
    }
