use crate::{definitions::Variable, gkr_compiler::graph::GKRGraph};

use super::*;

impl GKRGraph {
    pub(crate) fn print_nodes(&self) {
        for (node_idx, node) in self.all_nodes.iter().enumerate() {
            println!("Node {} is {}", node_idx, node.short_name());
        }
    }

    pub(crate) fn make_graphviz(&self, variable_names: &HashMap<Variable, String>) -> String {
        let mut node_lines = vec![];
        let mut edges = vec![];
        let mut visited_nodes = HashSet::new();

        let mut unique_edges = HashSet::new();
        for (var, pos) in self
            .base_layer_memory
            .iter()
            .chain(self.base_layer_witness.iter())
        {
            let node_idx = self.mapping[pos];
            let node = &self.all_nodes[node_idx];
            let deps = &self.dependencies[&node_idx];
            let short_name = node.short_name();
            let extra_name = variable_names.get(var).map(|el| el.as_str()).unwrap_or("");

            let node_descr = format!(
                "I{} [shape=ellipse,color=black,style=bold,label=\"{}({})\",labelloc=b];",
                node_idx, short_name, extra_name
            );
            node_lines.push(node_descr);

            for dep in deps.iter() {
                if unique_edges.contains(&(node_idx, dep)) == false {
                    let edge_descr = format!(
                        "I{} -> I{}  [style=bold,color=blue,arrowtail=normal,group=\"I{}\"]",
                        dep, node_idx, node_idx
                    );
                    edges.push(edge_descr);
                    unique_edges.insert((node_idx, dep));
                }
            }

            visited_nodes.insert(node_idx);
        }
        for (node_idx, node) in self.all_nodes.iter().enumerate() {
            if visited_nodes.contains(&node_idx) {
                continue;
            }
            let deps = &self.dependencies[&node_idx];
            let short_name = node.short_name();

            let node_descr = format!(
                "I{} [shape=box,color=black,style=bold,label=\"{}\",labelloc=b];",
                node_idx, short_name
            );
            node_lines.push(node_descr);

            for dep in deps.iter() {
                if unique_edges.contains(&(node_idx, dep)) == false {
                    let edge_descr = format!(
                        // "I{} -> I{}  [style=bold,color=blue,arrowtail=normal,group=\"I{}\"]",
                        "I{} -> I{}  [color=blue,arrowtail=normal,group=\"I{}\"]",
                        dep, node_idx, node_idx
                    );
                    edges.push(edge_descr);
                    unique_edges.insert((node_idx, dep));
                }
            }

            visited_nodes.insert(node_idx);
        }

        let all_nodes = node_lines.join("\n");
        let all_edges = edges.join("\n");

        let mut lines = vec![];
        lines.push("/* GKR graph */".to_string());
        lines.push("digraph G {{".to_string());

        lines.push("rankdir=\"LR\";".to_string());
        // lines.push("splines=true;".to_string());

        lines.push("graph [pad=\"1\", nodesep=\"4\", ranksep=\"64.0 equally\"];".to_string());
        // lines.push("graph [splines=ortho, pad=\"1\", nodesep=\"4\", ranksep=\"64.0 equally\"];".to_string());

        lines.push(all_nodes);
        lines.push(all_edges);
        lines.push("}}".to_string());

        let result = lines.join("\n");

        // let result = format!(
        //     "/* GKR graph */\n graph G {{\nrankdir=RL;\n{}\n{}\n}}\n",
        //     all_nodes, all_edges
        // );

        result

        // r##"/* ancestor graph from Caroline Bouvier Kennedy */
        // graph G {
        //     I5 [shape=ellipse,color=red,style=bold,label="Caroline Bouvier Kennedy\nb. 27.11.1957 New York",image="images/165px-Caroline_Kennedy.jpg",labelloc=b];
        //     I1 [shape=box,color=blue,style=bold,label="John Fitzgerald Kennedy\nb. 29.5.1917 Brookline\nd. 22.11.1963 Dallas",image="images/kennedyface.jpg",labelloc=b];
        //     I6 [shape=box,color=blue,style=bold,label="John Fitzgerald Kennedy\nb. 25.11.1960 Washington\nd. 16.7.1999 over the Atlantic Ocean, near Aquinnah, MA, USA",image="images/180px-JFKJr2.jpg",labelloc=b];
        //     I7 [shape=box,color=blue,style=bold,label="Patrick Bouvier Kennedy\nb. 7.8.1963\nd. 9.8.1963"];
        //     I2 [shape=ellipse,color=red,style=bold,label="Jaqueline Lee Bouvier\nb. 28.7.1929 Southampton\nd. 19.5.1994 New York City",image="images/jacqueline-kennedy-onassis.jpg",labelloc=b];
        //     I8 [shape=box,color=blue,style=bold,label="Joseph Patrick Kennedy\nb. 6.9.1888 East Boston\nd. 16.11.1969 Hyannis Port",image="images/1025901671.jpg",labelloc=b];
        //     I10 [shape=box,color=blue,style=bold,label="Joseph Patrick Kennedy Jr\nb. 1915\nd. 1944"];
        //     I11 [shape=ellipse,color=red,style=bold,label="Rosemary Kennedy\nb. 13.9.1918\nd. 7.1.2005",image="images/rosemary.jpg",labelloc=b];
        //     I12 [shape=ellipse,color=red,style=bold,label="Kathleen Kennedy\nb. 1920\nd. 1948"];
        //     I13 [shape=ellipse,color=red,style=bold,label="Eunice Mary Kennedy\nb. 10.7.1921 Brookline"];
        //     I9 [shape=ellipse,color=red,style=bold,label="Rose Elizabeth Fitzgerald\nb. 22.7.1890 Boston\nd. 22.1.1995 Hyannis Port",image="images/Rose_kennedy.JPG",labelloc=b];
        //     I15 [shape=box,color=blue,style=bold,label="Aristotle Onassis"];
        //     I3 [shape=box,color=blue,style=bold,label="John Vernou Bouvier III\nb. 1891\nd. 1957",image="images/BE037819.jpg",labelloc=b];
        //     I4 [shape=ellipse,color=red,style=bold,label="Janet Norton Lee\nb. 2.10.1877\nd. 3.1.1968",image="images/n48862003257_1275276_1366.jpg",labelloc=b];
        //      I1 -- I5  [style=bold,color=blue];
        //      I1 -- I6  [style=bold,color=orange];
        //      I2 -- I6  [style=bold,color=orange];
        //      I1 -- I7  [style=bold,color=orange];
        //      I2 -- I7  [style=bold,color=orange];
        //      I1 -- I2  [style=bold,color=violet];
        //      I8 -- I1  [style=bold,color=blue];
        //      I8 -- I10  [style=bold,color=orange];
        //      I9 -- I10  [style=bold,color=orange];
        //      I8 -- I11  [style=bold,color=orange];
        //      I9 -- I11  [style=bold,color=orange];
        //      I8 -- I12  [style=bold,color=orange];
        //      I9 -- I12  [style=bold,color=orange];
        //      I8 -- I13  [style=bold,color=orange];
        //      I9 -- I13  [style=bold,color=orange];
        //      I8 -- I9  [style=bold,color=violet];
        //      I9 -- I1  [style=bold,color=red];
        //      I2 -- I5  [style=bold,color=red];
        //      I2 -- I15  [style=bold,color=violet];
        //      I3 -- I2  [style=bold,color=blue];
        //      I3 -- I4  [style=bold,color=violet];
        //      I4 -- I2  [style=bold,color=red];
        //     }
        // "##
    }
}
