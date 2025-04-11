#include "SIPP.h"
#include <chrono>
#include <thread>

// Use experimental/filesystem for older compilers
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    // Fallback if filesystem is not available
    #define NO_FILESYSTEM
#endif

/* #########################
% Inputs:
%   - goal: A SIPP keresés végeredményeként kapott célcsomópont (LLNode*):
            + utolsó állapot
            + eddigi soft ütközések száma
%   - path: időindexelt útvonal célhelye.
%
% Function behaviour:
%   - A függvény a célcsomópontból (goal) indulva a szülő mutatók segítségével "visszaköveti" az útvonalat a gyökér csomópontig (ahol a timestep 0).
%   - globalis num_collision = célcsomópont num_of_conflicts értéke
%   - A path vektor mérete = célcsomópont időlépése +1
%   - Minden időlépésnél a megfelelő helyszínt (location) beállítja.
%   - Ha egy lépés között időköz van, akkor az ügynök "vár" az előző csomópont helyén.
%
% Outputs:
%   - A path vektor tartalmazza az ügynök teljes útvonalát az idő lépései szerint, a start állapottól (timestep 0) a célig.
*/
void SIPP::updatePath(const LLNode* goal, vector<PathEntry> &path)
{
    num_collisions = goal->num_of_conflicts;
	path.resize(goal->timestep + 1);
	// num_of_conflicts = goal->num_of_conflicts;

	const auto* curr = goal;
	while (curr->parent != nullptr) // non-root node
	{
		const auto* prev = curr->parent;
		int t = prev->timestep + 1;
		while (t < curr->timestep)
		{
			path[t].location = prev->location; // wait at prev location
			path[t].timestep = t; // set the timestep
			t++;
		}
		path[curr->timestep].location = curr->location; // move to curr location
		path[curr->timestep].timestep = curr->timestep; // set the timestep
		curr = prev;
	}
	assert(curr->timestep == 0);
	path[0].location = curr->location;
	path[0].timestep = 0; // set the timestep for the start node
}

/*#########################
// find path by A*
// Returns a path that minimizes the collisions with the paths in the path table, breaking ties by the length

% Inputs:
%   - constraint_table: többi ügynök (soft constraint) útvonalaiból származó korlátozások.
%
% Function behaviour:
%   1. reset() metódus: init() reservationtable
%   2. Lekéri a start hely első biztonságos intervallumát, ha foglalt visszatér egy üres úttal
%   3. Kiszámolja a holding_time-ot (a célcellában várni tudó időt) és a goalhoz tartozó utolsó ütközési időpontot.
%      Ezek alapján a kezdő heurisztika értékét határozza meg:
%         h = max{ my_heuristic(start), T_hold, T_last + 1 }
%      ahol $T_{last}$ a cél cellához tartozó utolsó ütközési idő.
%   4. Létrehoz egy új SIPPNode-t a start állapotban (location = start, timestep = 0, h, továbbá az intervallum adataival)
%   5. Egy iteratív ciklusban addig dolgozik, míg a focal_list nem üres:
%         a. Kiveszi a focal_list legjobb (legkisebb $f$-értékű) csomópontját, és növeli az expanzió számát.
%         b. Ha a kivett csomópont célcsomópont (curr->is_goal == true) -> visszakövetés —> return
%         c. Speciális eset kezelése:
%           Ha a csomópont helye megegyezik a célpozícióval, a csomópont nem vár a célban, és az aktuális idő >= T_hold:
%           - Lekéri a jövőbeni ütközések számát.
%           - Ha nincs további ütközés, azonnal visszaköveti az útvonalat.
%           - Ha van ütközés, akkor egy új, célcsomópontot hoz létre (is_goal = true, h = 0),
%            amelyhez hozzáadja a jövőbeni ütközések számát, majd dominancia ellenőrzés után a focal_list-be kerül.
%         d. Végigiterál az aktuális csomópont szomszédos celláin (instance.getNeighbors),
%            és minden szomszédra lekéri a \texttt{ReservationTable} által meghatározott biztonságos intervallumokat
%            a következő időléptéktől (curr->timestep + 1) a curr->high_expansion+1-ig.
%            Minden ilyen intervallum esetén:
%                - Ha a következő lépés ideje plusz a heurisztika meghaladja a constraint_table.length_max értékét, kilép.
%                - Kiszámolja az új csomópont soft ütközéseinek számát,
%                  illetve az új heurisztika értéket, amely a következő:
%                   (h_next) két érték maximuma:
%                       adott next_location-re: my_heuristic[next_location].
%                       Ha a következő állapotban ütközés (conflict) várható (next_collisions > 0), akkor T_hold értéke,
%                       Ellenkező esetben az aktuális csomópont teljes költségének (f_curr) értéke.
%                       biztosítva a várható ütközések kezelését a heurisztikai becslésben.
%         e. az ügynök a jelenlegi cellájában várhat is
%         f. rerun vizualizáció: a csomópontok és az útvonalak kirajzolása

% Outputs:
%   - Visszatér egy path vektorral, amely tartalmazza a megtalált, időindexelt, ütközésmentes útvonalat.
%
%#########################
*/
Path SIPP::findPath(const ConstraintTable& constraint_table)
{
    reset();
    ReservationTable reservation_table(constraint_table, goal_location);
    Path path;

    // Clear soft collision points for visualization
#if RERUN_ENABLED
    if (use_rerun) {
        clearSoftCollisionPoints();
    }
#endif

    Interval interval = reservation_table.get_first_safe_interval(start_location);
    if (get<0>(interval) > 0)
        return path;

    auto holding_time = constraint_table.getHoldingTime(constraint_table.length_min);
    auto last_target_collision_time = constraint_table.getLastCollisionTimestep(goal_location);
    auto h = max(max(my_heuristic[start_location], holding_time), last_target_collision_time + 1);

    auto start = new SIPPNode(start_location, 0, h, nullptr, 0, get<1>(interval), get<1>(interval),
                              get<2>(interval), get<2>(interval));
    pushNodeToFocal(start);

    int iteration = 0; // iteration counter for visualization
    const int MAX_ITERATIONS = 40000;

    // Clear the current best path for visualization
    current_best_path.clear();

    // Initialize visualization
#if RERUN_ENABLED
    if (use_rerun && rec) {
        // Visualize initial state with empty path
        rerunVisualization(0, path);
    }
#endif

    while (!focal_list.empty())  // algorithm 1
    {
        auto* curr = focal_list.top();  // top: best node in th heap
        focal_list.pop();  // remove top node
        curr->in_openlist = false;

        // Update visualization periodically
        if (iteration % 1 == 0) { // Update every 5 iterations for better visualization
#if RERUN_ENABLED
            if (use_rerun && rec) {
                // Create a temporary path to the current node
                Path temp_path;
                SIPPNode* temp_node = curr;
                while (temp_node != nullptr) {
                    temp_path.emplace_back(temp_node->location, temp_node->timestep);
                    temp_node = static_cast<SIPPNode*>(temp_node->parent);
                }
                std::reverse(temp_path.begin(), temp_path.end());

                // Log the current timestep for debugging
                std::cout << "[DEBUG] Current planning timestep: " << curr->timestep << " (iteration " << iteration << ")" << std::endl;

                // Visualize the current state
                rerunVisualization(iteration, temp_path);
            }
#endif
        }
        num_expanded++; // number of checked node
        assert(curr->location >= 0);

        // --- Extra Visualization: Log the current focal node with a distinct marker.
#if RERUN_ENABLED
        if (use_rerun) { // Always update the current node visualization
            int cx = curr->location % instance.num_of_cols;
            int cy = curr->location / instance.num_of_cols;
            std::vector<rerun::components::Position3D> current_node_vec;
            current_node_vec.push_back(rerun::components::Position3D(cx + 0.5f, cy + 0.5f, 0.4f)); // slightly higher z for visibility
            rec->log(log_prefix + "/sipp/current_node",
                    rerun::Points3D(rerun::Collection<rerun::components::Position3D>(current_node_vec))
                        .with_colors({rerun::components::Color(255, 0, 0)})  // bright red
                        .with_radii({0.5f})); // larger radius
        }
#endif

        // check if the popped node is a goal
        if (curr->is_goal)
        {
            updatePath(curr, path);
            break;
        }
        else if (curr->location == goal_location && // arrive at the goal location
                 !curr->wait_at_goal && // not wait at the goal location
                 curr->timestep >= holding_time) // the agent can hold the goal location afterward
        {
            int future_collisions = constraint_table.getFutureNumOfCollisions(curr->location, curr->timestep);
            if (future_collisions == 0)
            {
                updatePath(curr, path);
                break;
            }
            // generate a goal node
            auto goal = new SIPPNode(*curr);
            goal->is_goal = true;
            goal->h_val = 0;
            goal->num_of_conflicts += future_collisions;
            // try to retrieve it from the hash table
            if (dominanceCheck(goal))  // algorithm 3
                pushNodeToFocal(goal);
            else
                delete goal;
        }

        for (int next_location : instance.getNeighbors(curr->location)) // move to neighboring locations
        {
            // Dynamic obstacles are now soft constraints with increased cost
            // We'll handle them in the cost calculation below
            int next_time = curr->timestep + 1;
            int move_cost = instance.getMoveCost(curr->location, next_location, next_time);

            // Skip if the move is invalid (e.g., static obstacle)
            if (move_cost == INT_MAX) {
                continue;
            }

            // Check for dynamic obstacles (soft constraints) and mark them for visualization
#if RERUN_ENABLED
            if (use_rerun && instance.hasDynamicMapData()) {
                int next_row = instance.getRowCoordinate(next_location);
                int next_col = instance.getColCoordinate(next_location);

                // If there's a dynamic obstacle, mark it as a soft collision point
                if (next_time < instance.getNumTimeSteps() &&
                    instance.hasDynamicObstacle(next_time, next_row, next_col)) {
                    addSoftCollisionPoint(next_location);
                }
            }
#endif

            for (auto & i : reservation_table.get_safe_intervals(
                    curr->location, next_location, curr->timestep + 1, curr->high_expansion + 1)) // algorithm 2
            {
                int next_high_generation, next_timestep, next_high_expansion;
                bool next_v_collision, next_e_collision;
                tie(next_high_generation, next_timestep, next_high_expansion, next_v_collision, next_e_collision) = i;
                if (next_timestep + my_heuristic[next_location] > constraint_table.length_max)  //<upper_bound, low, high,  vertex collision, edge collision>
                    break;
                // Calculate the cost of this move, including dynamic obstacle cost
                int move_cost = instance.getMoveCost(curr->location, next_location, next_timestep);

                // Calculate conflicts and add dynamic obstacle cost
                auto next_collisions = curr->num_of_conflicts +
                                    // (int)curr->collision_v * max(next_timestep - curr->timestep - 1, 0) + // wait time
                                      (int)next_v_collision + (int)next_e_collision;

                // Calculate h-value
                auto next_h_val = max(my_heuristic[next_location], (next_collisions > 0?
                    holding_time : curr->getFVal()) - next_timestep); // path max

                // Generate node with the adjusted g-value (using move_cost instead of 1)
                auto next = new SIPPNode(next_location,
                                         curr->g_val + move_cost, // Use move_cost instead of just incrementing by 1
                                         next_h_val,
                                         curr,
                                         next_timestep,
                                         next_high_generation,
                                         next_high_expansion,
                                         next_v_collision,
                                         next_collisions);
                // try to retrieve it from the hash table
                if (dominanceCheck(next))
                    pushNodeToFocal(next);
                else
                    delete next;
            }
        }  // end for loop that generates successors
        // wait at the current location
        if (curr->high_expansion == curr->high_generation and
            reservation_table.find_safe_interval(interval, curr->location, curr->high_expansion) and
                get<0>(interval) + curr->h_val <= reservation_table.constraint_table.length_max) // up bound==t_max ,has net interval start with t_max, not exceed length limitation
        {
            auto next_timestep = get<0>(interval);  //==curr->high_expansion
            auto next_h_val = max(my_heuristic[curr->location], (get<2>(interval) ? holding_time : curr->getFVal()) - next_timestep);
            auto next_collisions = curr->num_of_conflicts +
                    // (int)curr->collision_v * max(next_timestep - curr->timestep - 1, 0) +
		    (int)get<2>(interval);
            auto next = new SIPPNode(curr->location, next_timestep, next_h_val, curr, next_timestep,
                                     get<1>(interval), get<1>(interval), get<2>(interval),
                                     next_collisions);
            next->wait_at_goal = (curr->location == goal_location);
            if (dominanceCheck(next))
                pushNodeToFocal(next);
            else
                delete next;
        }
#if RERUN_ENABLED
        if ((use_rerun) && iteration % 1 == 0) {
            std::vector<PathEntry> partial_path;
            const auto* trace = curr;
            while (trace) {
                PathEntry entry;
                entry.location = trace->location;
                entry.timestep = trace->timestep;
                partial_path.push_back(entry);
                trace = static_cast<const SIPPNode*>(trace->parent);
            }
            std::reverse(partial_path.begin(), partial_path.end());

            if (use_rerun) {
                // Log the current timestep for debugging
                if (!partial_path.empty()) {
                    std::cout << "[DEBUG] Partial path visualization with " << partial_path.size() << " steps, ending at timestep " << partial_path.back().timestep << " (iteration " << iteration << ")" << std::endl;
                } else {
                    std::cout << "[DEBUG] Empty partial path at iteration " << iteration << std::endl;
                }
                rerunVisualization(iteration, partial_path);

                // Also add a short sleep to allow the viewer to update.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
#endif
    }  // end while loop

    /*
    // At the end of the method, before returning:
    if (path.empty()) {
        std::cout << "SIPP::findPath - FAILED: No path found from " << start_location
                  << " to " << goal_location << std::endl;
    } else {
        std::cout << "SIPP::findPath - SUCCESS: Path found with "
                  << path.size() << " steps" << std::endl;
        std::cout << "  Final location: " << path.back().location << std::endl;
    }
    */

    //if (path.empty())
    //{
    //    printSearchTree();
    //}

    // Final visualization with the complete path
#if RERUN_ENABLED
    if (use_rerun && rec) {
        if (!path.empty()) {
            std::cout << "[DEBUG] Final path visualization with " << path.size() << " steps, ending at timestep " << path.back().timestep << " (final iteration " << iteration << ")" << std::endl;
            rerunVisualization(iteration, path);
        } else {
            std::cout << "[DEBUG] No path found, showing final state" << std::endl;
            // Even if no path was found, show the final state of the search
            rerunVisualization(iteration, path);
        }
    }
#endif

    releaseNodes();
    return path;
}

/*#########################
% SIPP::pushNodeToFocal
%
% Inputs:
%   - SIPPNode* node: Az újonnan generált csomópont, amelyet hozzá kell adni a fókuszlistához.
% Funkció:
%   - Növeli a globális generált csomópontok számát.
% Outputs:
%   - A csomópont bekerül a focal_list-be, és annak referenciája (handle) rögzítésre kerül.
*/
inline void SIPP::pushNodeToFocal(SIPPNode* node)
{
    num_generated++;
    allNodes_table[node].push_back(node);
    node->in_openlist = true;
    node->focal_handle = focal_list.push(node); // we only use focal list; no open list is used  // push add node to the heap and return its reference
}

/*#########################
% SIPP::eraseNodeFromLists
%
% Inputs:
%   - SIPPNode* node: A törlendő csomópont.
% Funkció:
%   - Sipp Nodeok eltávolítása a megfelelő listákból.
% Outputs:
%   - A csomópont eltávolításra kerül a megfelelő listákból.
*/
inline void SIPP::eraseNodeFromLists(SIPPNode* node)
{
    if (open_list.empty())
    { // we only have focal list
        focal_list.erase(node->focal_handle);
    }
    else if (focal_list.empty())
    {  // we only have open list
        open_list.erase(node->open_handle);
    }
    else
    { // we have both open and focal
        open_list.erase(node->open_handle);
        if (node->getFVal() <= w * min_f_val)
            focal_list.erase(node->focal_handle);
    }
}
/*#########################
% SIPP::releaseNodes
%
% Funkció:
%   - A csomópontok felszabadítása a memóriából.
% Outputs:
%   - A csomópontok törlésre kerülnek a memóriából.
*/
void SIPP::releaseNodes()
{
    open_list.clear();
    focal_list.clear();
    for (auto & node_list : allNodes_table)
        for (auto n : node_list.second)
            delete n;
    allNodes_table.clear();
    for (auto n : useless_nodes)
        delete n;
    useless_nodes.clear();
}

// return true iff we the new node is not dominated by any old node-- algorithm 3
/*
dominanceCheck(SIPPNode* new_node)
% Inputs:
%   - SIPPNode* new_node: Az új csomópont, amelyet ellenőrizni kell.
% Funkció:
%   - Ellenőrzi, hogy az új csomópont dominálja-e a már meglévő csomópontokat.
%   - Ha dominálja, akkor törli a régi csomópontokat.
% Outputs:
%   - Visszatér true-val, ha az új csomópont nem dominálja a régi csomópontokat.
*/
bool SIPP::dominanceCheck(SIPPNode* new_node)
{
    auto ptr = allNodes_table.find(new_node);  // hash table, one hash value can have multiple value
    if (ptr == allNodes_table.end())
        return true;
    for (auto & old_node : ptr->second)  // ptr->second: hash value
    {
        if (old_node->timestep <= new_node->timestep and
            old_node->num_of_conflicts <= new_node->num_of_conflicts)
        { // the new node is dominated by the old node, no need to generate new node
            return false;
        }
        else if (old_node->timestep >= new_node->timestep and
                old_node->num_of_conflicts >= new_node->num_of_conflicts) // the old node is dominated by the new node
        { // delete the old node
            if (old_node->in_openlist) // the old node has not been expanded yet
                eraseNodeFromLists(old_node); // delete it from open and/or focal lists
            else // the old node has been expanded already
                num_reopened++; //re-expand it
            useless_nodes.push_back(old_node);
            ptr->second.remove(old_node);
            num_generated--; // this is because we later will increase num_generated when we insert the new node into lists.
            return true;
        }
        else if(old_node->timestep < new_node->high_expansion and new_node->timestep < old_node->high_expansion)
        { // intervals overlap --> we need to split the node to make them disjoint
            if (old_node->timestep <= new_node->timestep)
            {
                assert(old_node->num_of_conflicts > new_node->num_of_conflicts);
                old_node->high_expansion = new_node->timestep;
            }
            else // i.e., old_node->timestep > new_node->timestep
            {
                assert(old_node->num_of_conflicts <= new_node->num_of_conflicts);
                new_node->high_expansion = old_node->timestep;
            }
        }
    }
    return true;
}


#if RERUN_ENABLED
void SIPP::rerunVisualization(int iteration, const Path& path)
{
    if (!use_rerun || !rec) return;

    int width  = instance.num_of_cols;
    int height = instance.num_of_rows;

    std::vector<rerun::components::Position3D> open_pts, closed_pts;

    // Get the current timestep from the path or use the last valid timestep
    int current_timestep = 0;

    // Store the last valid timestep across all iterations for consistent visualization
    static int global_last_valid_timestep = 0;

    // If we have a path, use the timestep from the last node in the path
    if (!path.empty()) {
        current_timestep = path.back().timestep;
        if (current_timestep > 0) {
            global_last_valid_timestep = current_timestep;
        }
    } else if (iteration > 0) {
        // If we don't have a path but we're in a later iteration, use the last valid timestep
        current_timestep = global_last_valid_timestep;
    }

    // Set the timeline for this visualization
    rec->set_time_sequence("planning_time", iteration);

    // Log dynamic obstacles for the current timestep
    if (instance.hasDynamicMapData()) {
        // First clear any previous dynamic obstacles to avoid overlapping
        rec->log(log_prefix + "/sipp/dynamic_obstacles", rerun::Clear());
        rec->log(log_prefix + "/sipp/timestep_indicator", rerun::Clear());

        // Make sure we don't exceed the available timesteps
        int num_timesteps = static_cast<int>(instance.getNumTimeSteps());

        // Use the exact timestep from the current node being expanded
        // This ensures dynamic obstacles are shown at the correct planning step
        int timestep_to_use = current_timestep;

        // Ensure timestep is valid
        if (timestep_to_use < 0) timestep_to_use = 0;
        if (timestep_to_use >= num_timesteps) timestep_to_use = num_timesteps - 1;

        // Always update the visualization at every iteration to ensure consistency
        // This ensures dynamic obstacles are always shown, even if the timestep hasn't changed

        // Create a separate entity for each timestep's dynamic obstacles
        // This way they're consistently visualized at each planning step
        std::vector<rerun::components::Position3D> dynamic_obstacle_pts;
        std::vector<rerun::components::Color> dynamic_obstacle_colors;
        std::vector<float> dynamic_obstacle_radii;

        // Count dynamic obstacles for debugging
        int obstacle_count = 0;

        // Iterate through the grid at the current timestep
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (instance.hasDynamicObstacle(timestep_to_use, y, x)) {
                    obstacle_count++;

                    // Position dynamic obstacles at a higher z-coordinate to ensure they're on top
                    dynamic_obstacle_pts.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.5f));

                    // Use a lighter green color (0, 200, 0) for better visibility
                    dynamic_obstacle_colors.push_back(rerun::components::Color(0, 200, 0, 255)); // Lighter green

                    // The size of dynamic obstacles is already set in mapf_gym.py
                    // We just need to ensure they're visible in the visualization
                    dynamic_obstacle_radii.push_back(0.5f); // Standard radius for visibility
                }
            }
        }

        // Add a visual indicator of the current timestep
        // Create a text annotation using a 3D point with a label
        std::vector<rerun::components::Position3D> timestep_pos;
        timestep_pos.push_back(rerun::components::Position3D(width / 2.0f, -1.0f, 1.0f));

        std::vector<std::string> timestep_labels;
        timestep_labels.push_back("Timestep: " + std::to_string(timestep_to_use));

        rec->log(log_prefix + "/sipp/timestep_indicator",
                rerun::Points3D(rerun::Collection<rerun::components::Position3D>(timestep_pos))
                    .with_colors({rerun::components::Color(255, 255, 255)})
                    .with_radii({0.1f})
                    .with_labels(timestep_labels));

        // Log dynamic obstacles with a lighter green color
        if (!dynamic_obstacle_pts.empty()) {
            std::cout << "[DEBUG] Visualizing " << obstacle_count << " dynamic obstacles at timestep " << timestep_to_use << " (iteration " << iteration << ")" << std::endl;

            // Set a specific timeline key for this visualization to ensure consistency
            rec->set_time_sequence("planning_time", iteration);

            // Clear previous dynamic obstacles
            rec->log(log_prefix + "/sipp/dynamic_obstacles", rerun::Clear());

            // Log the dynamic obstacles for this timestep
            rec->log(log_prefix + "/sipp/dynamic_obstacles",
                    rerun::Points3D(rerun::Collection<rerun::components::Position3D>(dynamic_obstacle_pts))
                        .with_colors(dynamic_obstacle_colors)
                        .with_radii(dynamic_obstacle_radii));

            // Add a visual trail to show the movement of dynamic obstacles
            // This helps visualize how they move over time
            static std::vector<rerun::components::Position3D> trail_pts;
            static std::vector<rerun::components::Color> trail_colors;
            static std::vector<float> trail_radii;
            static int last_timestep = -1;

            // Only update the trail when the timestep changes
            if (timestep_to_use != last_timestep) {
                // Add the current dynamic obstacles to the trail with a faded color
                for (size_t i = 0; i < dynamic_obstacle_pts.size(); i++) {
                    trail_pts.push_back(dynamic_obstacle_pts[i]);
                    // Use a more transparent green for the trail
                    trail_colors.push_back(rerun::components::Color(0, 200, 0, 100)); // Semi-transparent green
                    trail_radii.push_back(0.3f); // Smaller radius for trail points
                }

                // Limit the trail size to avoid performance issues
                const size_t max_trail_size = 1000;
                if (trail_pts.size() > max_trail_size) {
                    trail_pts.erase(trail_pts.begin(), trail_pts.begin() + (trail_pts.size() - max_trail_size));
                    trail_colors.erase(trail_colors.begin(), trail_colors.begin() + (trail_colors.size() - max_trail_size));
                    trail_radii.erase(trail_radii.begin(), trail_radii.begin() + (trail_radii.size() - max_trail_size));
                }

                last_timestep = timestep_to_use;
            }

            // Log the trail
            if (!trail_pts.empty()) {
                rec->log(log_prefix + "/sipp/dynamic_obstacles_trail",
                        rerun::Points3D(rerun::Collection<rerun::components::Position3D>(trail_pts))
                            .with_colors(trail_colors)
                            .with_radii(trail_radii));
            }
        } else {
            std::cout << "[WARNING] No dynamic obstacles found at timestep " << timestep_to_use << " (iteration " << iteration << ")" << std::endl;
        }
    }

    // Log agent start position (red)
    std::vector<rerun::components::Position3D> start_pos;
    int start_x = instance.getColCoordinate(start_location);
    int start_y = instance.getRowCoordinate(start_location);
    start_pos.push_back(rerun::components::Position3D(start_x + 0.5f, start_y + 0.5f, 0.3f));
    rec->log(log_prefix + "/sipp/agent_start",
             rerun::Points3D(rerun::Collection<rerun::components::Position3D>(start_pos))
                 .with_colors({rerun::components::Color(255, 105, 180)}) // Pink
                 .with_radii({0.4f}));

    // Log agent goal position (green)
    std::vector<rerun::components::Position3D> goal_pos;
    int goal_x = instance.getColCoordinate(goal_location);
    int goal_y = instance.getRowCoordinate(goal_location);
    goal_pos.push_back(rerun::components::Position3D(goal_x + 0.5f, goal_y + 0.5f, 0.3f));
    rec->log(log_prefix + "/sipp/agent_goal",
             rerun::Points3D(rerun::Collection<rerun::components::Position3D>(goal_pos))
                 .with_colors({rerun::components::Color(0, 0, 255)}) // Blue
                 .with_radii({0.4f}));

    // Log other agents' start positions (brown)
    if (!other_agents_starts.empty()) {
        std::vector<rerun::components::Position3D> other_starts;
        std::vector<rerun::components::Color> other_start_colors;

        for (const auto& loc : other_agents_starts) {
            int x = instance.getColCoordinate(loc);
            int y = instance.getRowCoordinate(loc);
            other_starts.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.3f));
            other_start_colors.push_back(rerun::components::Color(255, 105, 180)); // Pink
        }

        rec->log(log_prefix + "/sipp/other_starts",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(other_starts))
                     .with_colors(other_start_colors)
                     .with_radii(std::vector<float>(other_starts.size(), 0.3f)));
    }

    // Log other agents' goal positions (pink)
    if (!other_agents_goals.empty()) {
        std::vector<rerun::components::Position3D> other_goals;
        std::vector<rerun::components::Color> other_goal_colors;

        for (const auto& loc : other_agents_goals) {
            int x = instance.getColCoordinate(loc);
            int y = instance.getRowCoordinate(loc);
            other_goals.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.3f));
            other_goal_colors.push_back(rerun::components::Color(165, 42, 42)); // Brown
        }

        rec->log(log_prefix + "/sipp/other_goals",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(other_goals))
                     .with_colors(other_goal_colors)
                     .with_radii(std::vector<float>(other_goals.size(), 0.3f)));
    }

    // Other agents' paths (light blue) - Added in the previous section

    // Log soft collision points (red X markers)
    if (!soft_collision_points.empty()) {
        std::vector<rerun::components::Position3D> collision_points;
        std::vector<rerun::components::Color> collision_colors;

        for (const auto& loc : soft_collision_points) {
            int x = instance.getColCoordinate(loc);
            int y = instance.getRowCoordinate(loc);
            collision_points.push_back(rerun::components::Position3D(x + 0.5f, y + 0.5f, 0.4f)); // Higher z to be visible
            collision_colors.push_back(rerun::components::Color(255, 0, 0)); // Red for collisions
        }

        // Log collision points as red circles
        rec->log(log_prefix + "/sipp/soft_collisions",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(collision_points))
                     .with_colors(collision_colors)
                     .with_radii(std::vector<float>(collision_points.size(), 0.3f)));

        // Create X markers for each collision point
        for (size_t i = 0; i < collision_points.size(); i++) {
            const auto& pos = collision_points[i];
            float x = pos.x();
            float y = pos.y();
            float z = pos.z() + 0.01f; // Slightly above the collision point
            float size = 0.4f; // Size of the X

            // Create two line segments for the X
            std::vector<rerun::components::Position3D> line1 = {
                rerun::components::Position3D(x - size/2, y - size/2, z),
                rerun::components::Position3D(x + size/2, y + size/2, z)
            };

            std::vector<rerun::components::Position3D> line2 = {
                rerun::components::Position3D(x - size/2, y + size/2, z),
                rerun::components::Position3D(x + size/2, y - size/2, z)
            };

            std::vector<rerun::components::LineStrip3D> strips1 = {
                rerun::components::LineStrip3D(
                    rerun::Collection<rerun::components::Position3D>(line1)
                )
            };

            std::vector<rerun::components::LineStrip3D> strips2 = {
                rerun::components::LineStrip3D(
                    rerun::Collection<rerun::components::Position3D>(line2)
                )
            };

            // Log the X marker
            rec->log(log_prefix + "/sipp/soft_collisions/x_marker_" + std::to_string(i) + "_1",
                     rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(strips1))
                         .with_colors({rerun::components::Color(255, 0, 0)}) // Red
                         .with_radii({0.1f}));

            rec->log(log_prefix + "/sipp/soft_collisions/x_marker_" + std::to_string(i) + "_2",
                     rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(strips2))
                         .with_colors({rerun::components::Color(255, 0, 0)}) // Red
                         .with_radii({0.1f}));
        }
    }

    // Log node sets (open and closed) using the same coordinate system.
    for (const auto& pair : allNodes_table) {
        for (const auto& node : pair.second) {
            int nx = node->location % width;
            int ny = node->location / width;
            rerun::components::Position3D pos(nx + 0.5f, ny + 0.5f, 0.0f);
            if (node->in_openlist)
                open_pts.push_back(pos);
            else
                closed_pts.push_back(pos);
        }
    }

    // Log open nodes (green)
    if (!open_pts.empty()) {
        rec->log(log_prefix + "/sipp/open_nodes",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(open_pts))
                     .with_colors(std::vector<rerun::components::Color>(open_pts.size(), rerun::components::Color(0, 255, 0)))
                     .with_radii(std::vector<float>(open_pts.size(), 0.2f)));
    }

    // Log closed nodes (yellow)
    if (!closed_pts.empty()) {
        rec->log(log_prefix + "/sipp/closed_nodes",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(closed_pts))
                     .with_colors(std::vector<rerun::components::Color>(closed_pts.size(), rerun::components::Color(255, 255, 0)))
                     .with_radii(std::vector<float>(closed_pts.size(), 0.2f)));
    }

    // Log the current path (orange)
    if (!path.empty()) {
        std::vector<rerun::components::Position3D> path_positions;
        for (const auto& step : path) {
            int px = instance.getColCoordinate(step.location);
            int py = instance.getRowCoordinate(step.location);
            path_positions.push_back(rerun::components::Position3D(px + 0.5f, py + 0.5f, 0.25f));
        }

        // Create a line strip for the path
        if (path_positions.size() > 1) {
            std::vector<rerun::components::LineStrip3D> strips = {
                rerun::components::LineStrip3D(
                    rerun::Collection<rerun::components::Position3D>(path_positions)
                )
            };
            rec->log(log_prefix + "/sipp/path_line",
                     rerun::LineStrips3D(rerun::Collection<rerun::components::LineStrip3D>(strips))
                         .with_colors({rerun::components::Color(255, 165, 0)}) // Orange
                         .with_radii({0.15f}));
        }

        // Also log the points for better visibility
        rec->log(log_prefix + "/sipp/path_points",
                 rerun::Points3D(rerun::Collection<rerun::components::Position3D>(path_positions))
                     .with_colors(std::vector<rerun::components::Color>(path_positions.size(), rerun::components::Color(255, 165, 0))) // Orange
                     .with_radii(std::vector<float>(path_positions.size(), 0.3f)));
    }
}
#else
void SIPP::rerunVisualization(int iteration, const Path& path)
{
    // Empty implementation when Rerun is disabled
}
#endif