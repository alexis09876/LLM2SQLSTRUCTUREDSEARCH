import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from backend.src.data_io.file_reader import FileReader
from backend.src.data_io.file_writer import FileWriter
from backend.src.schema_parser.column_record import ColumnRecord
from backend.src.schema_parser.edge_record import EdgeRecord

class DevSchemaCatalog:
    """
    Build a column catalog and explicit FK edges from dev_tables.json.
    - No augmented edges here.
    - Ready for two-step search pipelines (entity->column + join path).
    """

    def __init__(self, dev_data: List[Dict[str, Any]]):
        self._raw: List[Dict[str, Any]] = dev_data
        self.column_catalog: List[ColumnRecord] = []
        self.edges: List[EdgeRecord] = []
        self._build()

    # ---------- Constructors ----------
    @classmethod
    def from_dev_json(cls, path: str) -> "DevSchemaCatalog":
        data = FileReader.load_dev_tables(path)
        if data is None:
            raise ValueError(f"Failed to load dev tables from {path}")
        return cls(data)

    # ---------- Public helpers ----------
    def db_ids(self) -> List[str]:
        return [db["db_id"] for db in self._raw]

    def get_db(self, db_id: str) -> Optional[Dict[str, Any]]:
        for db in self._raw:
            if db.get("db_id") == db_id:
                return db
        return None

    def get_tables(self, db_id: str) -> List[str]:
        db = self.get_db(db_id)
        return db["table_names_original"][:] if db else []

    def get_columns_of_table(self, db_id: str, table_name: str) -> List[ColumnRecord]:
        return [r for r in self.column_catalog if r.db_id == db_id and r.table == table_name]

    def as_catalog_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.column_catalog]

    def as_edges_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.edges]

    # ---------- Internal build ----------
    def _build(self) -> None:
        for db in self._raw:
            self._build_one_db(db)

    def _build_one_db(self, db: Dict[str, Any]) -> None:
        db_id = db["db_id"]
        table_names_orig: List[str] = db["table_names_original"]
        table_names_human: List[str] = db.get("table_names", table_names_orig)

        column_names_orig: List[List[Any]] = db["column_names_original"]  # [ [table_idx, colname], ... ]
        column_names_human: List[List[Any]] = db.get("column_names", column_names_orig)
        column_types: List[str] = db["column_types"]

        # normalize primary keys to groups: [[cid], [cid], [cid1,cid2], ...]
        raw_pks = db.get("primary_keys", [])
        pk_groups: List[List[int]] = []
        for item in raw_pks:
            pk_groups.append(item if isinstance(item, list) else [item])

        colid_to_pk_group: Dict[int, Optional[str]] = {}
        for gi, group in enumerate(pk_groups):
            gid = f"PKG_{db_id}_{gi+1}"
            for cid in group:
                colid_to_pk_group[cid] = gid

        fks: List[List[int]] = db.get("foreign_keys", [])

        def col_tuple(col_id: int) -> Tuple[str, str]:
            t_idx, colname = column_names_orig[col_id]
            if t_idx == -1:
                return ("*", "*")
            return (table_names_orig[t_idx], colname)

        # edges: explicit FK only
        for child_cid, parent_cid in fks:
            c_table, c_col = col_tuple(child_cid)
            p_table, p_col = col_tuple(parent_cid)
            if c_table != "*" and p_table != "*":
                self.edges.append(EdgeRecord(
                    db_id=db_id,
                    child_table=c_table, child_column=c_col, child_col_id=child_cid,
                    parent_table=p_table, parent_column=p_col, parent_col_id=parent_cid
                ))

        # ordinals within each table
        table_to_cids: Dict[str, List[int]] = {}
        for cid, (t_idx, _) in enumerate(column_names_orig):
            if t_idx == -1:
                continue
            tname = table_names_orig[t_idx]
            table_to_cids.setdefault(tname, []).append(cid)

        ordinal_map: Dict[int, int] = {}
        for tname, cids in table_to_cids.items():
            for ord_i, cid in enumerate(cids):
                ordinal_map[cid] = ord_i

        # fk child -> parent mapping
        fk_child_to_parent: Dict[int, Tuple[str, str, int]] = {}
        for child_cid, parent_cid in fks:
            pt, pc = col_tuple(parent_cid)
            fk_child_to_parent[child_cid] = (pt, pc, parent_cid)

        # build column records (skip index 0 which is usually "*")
        for cid in range(1, len(column_names_orig)):
            t_idx, colname_orig = column_names_orig[cid]
            if t_idx == -1:
                continue

            tname_orig = table_names_orig[t_idx]
            tname_human = table_names_human[t_idx]
            _, colname_human = column_names_human[cid]
            ctype = column_types[cid]

            is_pk = cid in colid_to_pk_group
            pk_gid = colid_to_pk_group.get(cid)

            is_fk = cid in fk_child_to_parent
            fk_ref_table = fk_ref_column = None
            fk_ref_col_id = None
            fk_conf = None
            if is_fk:
                fk_ref_table, fk_ref_column, fk_ref_col_id = fk_child_to_parent[cid]
                fk_conf = 1.0

            aliases = list(dict.fromkeys([
                colname_orig,
                colname_human,
                f"{tname_orig}.{colname_orig}",
                f"{tname_human}.{colname_human}",
            ]))

            self.column_catalog.append(ColumnRecord(
                db_id=db_id,
                table=tname_orig,
                column=colname_orig,
                col_id=cid,
                ordinal_in_table=ordinal_map.get(cid),
                type=ctype,

                is_pk=is_pk,
                pk_group_id=pk_gid,
                is_fk=is_fk,
                fk_ref_table=fk_ref_table,
                fk_ref_column=fk_ref_column,
                fk_ref_col_id=fk_ref_col_id,
                fk_confidence=fk_conf,

                table_desc=None,
                column_desc=None,
                aliases=aliases
            ))

    def save_outputs(self, out_dir: str) -> None:
        """
        Save column catalog and edge list into JSON files.

        Args:
            out_dir (str): The directory where the output JSON files will be saved.

        Returns:
            None
        """
        cols = self.as_catalog_dicts()
        edges = self.as_edges_dicts()

        FileWriter.write_json(cols, os.path.join(out_dir, "columns.json"))
        FileWriter.write_json(edges, os.path.join(out_dir, "edges.json"))


if __name__ == "__main__":
    dev_json_path = "data/raw/data_minidev/MINIDEV/dev_tables.json"
    out_dir = "backend/src/schema_parser/output"

    catalog = DevSchemaCatalog.from_dev_json(dev_json_path)
    catalog.save_outputs(out_dir)
