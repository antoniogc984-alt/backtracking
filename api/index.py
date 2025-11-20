from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== MODELOS Pydantic =====================

class Edge(BaseModel):
    # El JSON que manda el front trae "from" y "to"
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")

    class Config:
        allow_population_by_field_name = True


class SolveRequest(BaseModel):
    nodes: List[str]
    edges: List[Edge]
    num_colors: int = 3


class Step(BaseModel):
    assignment: Dict[str, int]
    note: str


class SolveResponse(BaseModel):
    steps: List[Step]


# ===================== APP FASTAPI =====================

app = FastAPI(title="Coloreo de mapas - Backtracking API")

# CORS: permitir TODOS los orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 🔓 permitir cualquier origen
    allow_credentials=False,   # con "*" debe ser False
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API de backtracking funcionando 🚀"}


# ===================== LÓGICA DE BACKTRACKING =====================

def build_adjacency(nodes: List[str], edges: List[Edge]) -> Dict[str, List[str]]:
    """Construye lista de adyacencia a partir de la lista de aristas."""
    adj = {node: [] for node in nodes}
    for e in edges:
        if e.from_node in adj and e.to_node in adj:
            adj[e.from_node].append(e.to_node)
            adj[e.to_node].append(e.from_node)
    return adj


def is_valid_color(
    node: str,
    color: int,
    assignment: Dict[str, int],
    adjacency: Dict[str, List[str]]
) -> bool:
    """Revisa que ningún vecino del nodo tenga el mismo color."""
    for neighbor in adjacency.get(node, []):
        if assignment.get(neighbor) == color:
            return False
    return True


def solve_backtracking(
    nodes: List[str],
    adjacency: Dict[str, List[str]],
    num_colors: int,
) -> List[Step]:
    """
    Backtracking para coloreo de grafos.
    Se queda SOLO con la primera solución encontrada.
    """
    steps: List[Step] = []
    assignment: Dict[str, int] = {}

    def backtrack(index: int) -> bool:
        # Si ya asignamos color a todos los nodos → solución completa
        if index == len(nodes):
            steps.append(
                Step(
                    assignment=dict(assignment),
                    note="✅ Solución completa encontrada."
                )
            )
            # Detenernos en la primera solución
            return True

        node = nodes[index]

        for color in range(num_colors):
            # 1) Intentar asignar este color al nodo
            assignment[node] = color
            steps.append(
                Step(
                    assignment=dict(assignment),
                    note=f"Probando color {color} para el nodo {node}."
                )
            )

            # 2) Verificar si es válido con respecto a sus vecinos
            if is_valid_color(node, color, assignment, adjacency):
                # Continuar con el siguiente nodo
                if backtrack(index + 1):
                    # Si desde abajo nos dicen que ya hay solución, subimos sin más cambios
                    return True
            else:
                # Hay conflicto
                steps.append(
                    Step(
                        assignment=dict(assignment),
                        note=f"❌ Conflicto: el nodo {node} con color {color} "
                             f"choca con alguno de sus vecinos."
                    )
                )

            # 3) Backtracking: deshacer la asignación (solo si todavía no hay solución)
            assignment.pop(node, None)
            steps.append(
                Step(
                    assignment=dict(assignment),
                    note=f"↩ Backtracking: quitando color al nodo {node}."
                )
            )

        # No se encontró solución con ninguna combinación para este nodo
        return False

    backtrack(0)
    return steps


# ===================== ENDPOINT PRINCIPAL =====================

@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    """
    Endpoint principal:
      - Recibe nodos, aristas y número de colores.
      - Ejecuta el backtracking.
      - Devuelve la lista de pasos para la ANIMACIÓN.
    """
    if not request.nodes:
        return SolveResponse(steps=[])

    adjacency = build_adjacency(request.nodes, request.edges)
    steps = solve_backtracking(request.nodes, adjacency, request.num_colors)
    return SolveResponse(steps=steps)
